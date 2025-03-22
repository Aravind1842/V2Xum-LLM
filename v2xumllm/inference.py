import os
import sys
import random
import re
import cv2
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
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip
import matplotlib.pyplot as plt
import seaborn as sns
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


def visualize_frames(raw_frames, transformed_frames):
    """
    Display 4 random frames before and after transformation
    """
    plt.figure(figsize=(20, 10))
    
    # Randomly select 4 frame indices
    num_frames = raw_frames.shape[0]
    indices = random.sample(range(num_frames), min(4, num_frames))
    
    for i, idx in enumerate(indices):
        # Original frame
        plt.subplot(2, 4, i+1)
        # Convert from tensor [C,H,W] to numpy [H,W,C] and from RGB to BGR for display
        raw_frame = raw_frames[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(raw_frame)
        plt.title(f"Original Frame {idx}")
        plt.axis('off')
        
        # Transformed frame
        plt.subplot(2, 4, i+5)
        # Denormalize and convert to numpy for display
        trans_frame = transformed_frames[idx].permute(1, 2, 0).cpu().numpy()
        # Denormalize using the normalization values
        trans_frame = trans_frame * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
        trans_frame = np.clip(trans_frame, 0, 1)
        plt.imshow(trans_frame)
        plt.title(f"Transformed Frame {idx}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("frame_visualization.png")
    plt.show()


def visualize_feature_vector(features):
    """
    Create a visualization of the 768D feature vector
    """
    # 1. Select a random frame
    frame_idx = random.randint(0, features.shape[0]-1)
    feature_vector = features[frame_idx].cpu().numpy()
    
    # 2. Apply PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    # Reshape to include other frames as context for PCA
    all_features = features.cpu().numpy()
    pca.fit(all_features)
    feature_2d = pca.transform(all_features)
    
    # 3. Create visualizations
    plt.figure(figsize=(16, 12))
    
    # 3.1 Heatmap of the first 100 elements
    plt.subplot(2, 2, 1)
    sns.heatmap(feature_vector[:100].reshape(10, 10), cmap='viridis', annot=False)
    plt.title(f"Heatmap of First 100 Features (Frame {frame_idx})")
    
    # 3.2 Histogram of all values
    plt.subplot(2, 2, 2)
    plt.hist(feature_vector, bins=50)
    plt.title(f"Distribution of Feature Values (Frame {frame_idx})")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    
    # 3.3 Feature Magnitude (L2 norm per dimension)
    plt.subplot(2, 2, 3)
    magnitudes = np.abs(feature_vector)
    plt.plot(magnitudes)
    plt.title(f"Feature Magnitude per Dimension (Frame {frame_idx})")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Absolute Magnitude")
    
    # 3.4 PCA visualization - scatter plot of all frames with the selected frame highlighted
    plt.subplot(2, 2, 4)
    plt.scatter(feature_2d[:, 0], feature_2d[:, 1], alpha=0.5, label="All Frames")
    plt.scatter(feature_2d[frame_idx, 0], feature_2d[frame_idx, 1], color='red', s=100, label=f"Frame {frame_idx}")
    plt.title("PCA of Frame Features (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("feature_visualization.png")
    plt.show()
    
    print(f"Feature vector properties (Frame {frame_idx}):")
    print(f"  Shape: {feature_vector.shape}")
    print(f"  Min value: {feature_vector.min():.4f}")
    print(f"  Max value: {feature_vector.max():.4f}")
    print(f"  Mean: {feature_vector.mean():.4f}")
    print(f"  Std dev: {feature_vector.std():.4f}")


def visualize_logits(logits, tokenizer):
    """
    Visualize the model's output logits
    """
    # Choose a random position in the sequence
    logit_step = random.randint(0, len(logits)-1)
    selected_logits = logits[logit_step][0].cpu().float()
    
    # Get the top predicted tokens
    top_k = 20
    top_values, top_indices = torch.topk(selected_logits, top_k)
    
    # Convert to probabilities
    probs = torch.softmax(selected_logits, dim=0)
    top_probs = probs[top_indices]
    
    # Get token strings
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot top tokens and their probabilities
    plt.subplot(2, 1, 1)
    plt.bar(range(top_k), top_probs.numpy())
    plt.xticks(range(top_k), top_tokens, rotation=45, ha='right')
    plt.title(f"Top {top_k} Token Probabilities at Step {logit_step}")
    plt.xlabel("Token")
    plt.ylabel("Probability")
    
    # Plot logit value distribution
    plt.subplot(2, 1, 2)
    plt.hist(selected_logits.numpy(), bins=50)
    plt.title(f"Distribution of Logit Values at Step {logit_step}")
    plt.xlabel("Logit Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("logits_visualization.png")
    plt.show()
    
    print(f"Logits properties at step {logit_step}:")
    print(f"  Vocabulary size: {selected_logits.shape[0]}")
    print(f"  Min logit: {selected_logits.min().item():.4f}")
    print(f"  Max logit: {selected_logits.max().item():.4f}")
    print(f"  Most likely next token: '{tokenizer.decode([top_indices[0].item()])}'")
    print(f"  With probability: {top_probs[0].item():.4f}")


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

    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS of video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
    duration = int(num_frames / fps)  # Duration in seconds
    cap.release()

    video_loader = VideoExtractor(N=duration)
    _, raw_images = video_loader.extract({'id': None, 'video': args.video_path})

    # Display frames before transformation
    print("\n===== DISPLAYING ORIGINAL AND TRANSFORMED FRAMES =====")
    
    # Save copy of original frames for visualization
    original_frames = raw_images.clone()
    
    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Print image shape before transformation
    print("Original Video Frames Shape:", raw_images.shape)  # <N, 3, H, W>
    
    # Transform images
    transformed_images = transform(raw_images / 255.0)
    transformed_images = transformed_images.to(torch.float16)
    
    # Visualize frames before and after transformation
    visualize_frames(original_frames, transformed_images)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(transformed_images.to('cuda'))

    print("\n===== FEATURE EXTRACTION RESULTS =====") 
    # Print feature information
    print("Encoder Image Features Shape:", features.shape)
    print("Encoder Image Features Sample (first 10 elements of first frame):", features[0, :10].tolist())
    
    # Visualize feature vector
    visualize_feature_vector(features)

    prompts = {
        "V-sum": ["Please generate a VIDEO summarization for this video."],
        "T-sum": ["Please generate a TEXT summarization for this video."],
        "VT-sum": ["Please generate BOTH video and text summarization for this video."]
    }

    query = random.choice(prompts["VT-sum"])
    text_summary, keyframes, logits = inference(model, features, "<video>\n " + query, tokenizer)

    print("\n===== MODEL OUTPUT RESULTS =====") 
    
    print("\nText Summary:", text_summary)
    
    print("\nKeyframes Identified:")
    if keyframes:
        for i, keyframe in enumerate(keyframes):
            print(f"Segment {i+1}: {keyframe}")
    else:
        print("No keyframes were identified in the output.")
    
    # Visualize logits
    print("\n===== MODEL LOGITS VISUALIZATION =====")
    visualize_logits(logits, tokenizer)
