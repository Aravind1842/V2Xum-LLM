import os
import sys
import random
import re
import argparse
import torch
from v2xumllm.constants import IMAGE_TOKEN_INDEX
from v2xumllm.conversation import conv_templates, SeparatorStyle
from v2xumllm.model.builder import load_pretrained_model, load_lora
from v2xumllm.utils import disable_torch_init
from v2xumllm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToPILImage
import numpy as np
import clip
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def clean_output(text):
    # This regex removes anything between square brackets including the brackets
    return re.sub(r'\s*\[[^\]]*\]', '', text)


def extract_keyframes(text):
    # Extract content within square brackets
    keyframes = re.findall(r'\[([^\]]*)\]', text)
    return keyframes


def inference(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
            output_scores=True,  # Request output scores (logits)
            return_dict_in_generate=True  # Return a dictionary containing logits
        )

        logits = outputs.scores  # Extract logits

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != outputs.sequences[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    decoded_outputs = tokenizer.batch_decode(outputs.sequences[:, input_token_len:], skip_special_tokens=True)[0]
    
    # Get the original output with brackets
    original_output = decoded_outputs.strip()
    if original_output.endswith(stop_str):
        original_output = original_output[:-len(stop_str)]
    original_output = original_output.strip()
    
    # Extract keyframes before cleaning
    keyframes = extract_keyframes(original_output)
    
    # Clean the output to remove content between square brackets
    cleaned_output = clean_output(original_output)
    
    return cleaned_output, keyframes, logits


def visualize_features(features):
    """Visualize high-dimensional features using PCA and a heatmap."""
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_np = features.cpu().numpy()
    pca_result = pca.fit_transform(features_np)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PCA visualization of feature vectors
    scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=range(len(pca_result)), cmap='viridis', 
                         alpha=0.7)
    ax1.set_title('PCA of 768D Feature Vectors')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    fig.colorbar(scatter, ax=ax1, label='Frame Index')
    
    # Plot 2: Heatmap of feature activations
    # Take a sample of frames and dimensions for the heatmap to avoid overcrowding
    sample_size = min(10, features.shape[0])
    feature_sample = features[:sample_size, :50].cpu().numpy()  # Sample first 50 dimensions
    
    im = ax2.imshow(feature_sample, aspect='auto', cmap='plasma')
    ax2.set_title('Feature Activation Heatmap (First 50 dimensions)')
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Frame Index')
    fig.colorbar(im, ax=ax2, label='Activation Value')
    
    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.close()
    
    print("Feature visualization saved as 'feature_visualization.png'")


def display_sample_frames(original_frames, num_frames=3):
    """Display sample frames from the video."""
    # Select frames at equal intervals
    frame_indices = np.linspace(0, len(original_frames)-1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    
    to_pil = ToPILImage()
    
    for i, idx in enumerate(frame_indices):
        # Convert tensor to PIL image
        frame = original_frames[idx].cpu()
        frame = frame.mul(255).byte()  # Scale back to 0-255
        frame_pil = to_pil(frame)
        
        axes[i].imshow(frame_pil)
        axes[i].set_title(f'Frame {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_frames.png')
    plt.close()
    
    print("Sample frames saved as 'sample_frames.png'")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="/content/V2Xum-LLM-Models/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="/content/V2Xum-LLM-Models/llava-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="/content/V2Xum-LLM-Models/v2xumllm-vicuna-v1-5-7b-stage2-e2")
    parser.add_argument("--video_path", type=str, default="demo/Ex1.mp4")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)
    _, raw_images = video_loader.extract({'id': None, 'video': args.video_path})

    # Display sample frames from original video
    display_sample_frames(raw_images, num_frames=3)

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Print image shape before transformation
    print("Original Video Frames Shape:", raw_images.shape)  # <N, 3, H, W>
    
    # Transform images
    images = transform(raw_images / 255.0)
    images = images.to(torch.float16)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))
    
    # Print feature information
    print("Encoder Image Features Shape:", features.shape)
    print("Encoder Image Features Sample (first 10 elements of first frame):", features[0, :10].tolist())
    
    # Visualize the features
    visualize_features(features)

    prompts = {
        "V-sum": ["Please generate a VIDEO summarization for this video."],
        "T-sum": ["Please generate a TEXT summarization for this video."],
        "VT-sum": ["Please generate BOTH video and text summarization for this video."]
    }

    query = random.choice(prompts["VT-sum"])
    text_summary, keyframes, _ = inference(model, features, "<video>\n " + query, tokenizer)
    
    print("\nText Summary:", text_summary)
    
    print("\nKeyframes Identified:")
    if keyframes:
        for i, keyframe in enumerate(keyframes):
            print(f"Segment {i+1}: {keyframe}")
    else:
        print("No keyframes were identified in the output.")
