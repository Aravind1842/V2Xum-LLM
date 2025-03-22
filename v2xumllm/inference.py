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


def save_sample_frames(raw_frames, transformed_frames):
    """
    Save 4 random frames before and after transformation
    """
    os.makedirs("frames", exist_ok=True)
    
    # Randomly select 4 frame indices
    num_frames = raw_frames.shape[0]
    indices = random.sample(range(num_frames), min(4, num_frames))
    
    for i, idx in enumerate(indices):
        # Original frame
        raw_frame = raw_frames[idx].permute(1, 2, 0).cpu().numpy()
        raw_frame = (raw_frame * 255).astype(np.uint8)
        
        # OpenCV expects BGR, but our tensor is RGB, so convert
        raw_frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"frames/original_frame_{idx}.jpg", raw_frame_bgr)
        
        # Transformed frame - denormalize first
        trans_frame = transformed_frames[idx].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        trans_frame = trans_frame * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
        trans_frame = np.clip(trans_frame, 0, 1)
        trans_frame = (trans_frame * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        trans_frame_bgr = cv2.cvtColor(trans_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"frames/transformed_frame_{idx}.jpg", trans_frame_bgr)
    
    print(f"Saved {len(indices)} original and transformed frames to the 'frames' directory")
    print(f"Frames saved: {indices}")


def analyze_feature_vector(features):
    """
    Create a text-based analysis of the feature vector
    """
    # Select a random frame
    frame_idx = random.randint(0, features.shape[0]-1)
    feature_vector = features[frame_idx].cpu().numpy()
    
    # Basic statistics
    print(f"\nFeature vector analysis for Frame {frame_idx}:")
    print(f"  Shape: {feature_vector.shape}")
    print(f"  Min value: {feature_vector.min():.4f}")
    print(f"  Max value: {feature_vector.max():.4f}")
    print(f"  Mean: {feature_vector.mean():.4f}")
    print(f"  Std dev: {feature_vector.std():.4f}")
    
    # Histogram-like representation of feature distribution
    bins = 10
    hist, bin_edges = np.histogram(feature_vector, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print("\nFeature value distribution (text histogram):")
    max_count = max(hist)
    scale_factor = 50 / max_count if max_count > 0 else 1
    
    for i in range(bins):
        bar_length = int(hist[i] * scale_factor)
        bar = '#' * bar_length
        print(f"  [{bin_centers[i]:.2f}]: {bar} ({hist[i]} values)")
    
    # Find most active dimensions
    top_indices = np.argsort(np.abs(feature_vector))[-10:][::-1]
    print("\nTop 10 most active dimensions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Dimension {idx}: {feature_vector[idx]:.4f}")
    
    # Save feature vector to file
    np.savetxt(f"feature_vector_frame_{frame_idx}.txt", feature_vector)
    print(f"Full feature vector saved to feature_vector_frame_{frame_idx}.txt")

    # Calculate similarity to other frames
    if features.shape[0] > 1:
        print("\nSimilarity to other frames:")
        # Normalize the vectors for cosine similarity
        normalized_features = features.cpu().numpy() / np.linalg.norm(features.cpu().numpy(), axis=1, keepdims=True)
        selected_feature_norm = normalized_features[frame_idx]
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_features, selected_feature_norm)
        
        # Find most and least similar frames (excluding self)
        similarities[frame_idx] = -float('inf')  # Exclude self
        most_similar_idx = np.argmax(similarities)
        similarities[frame_idx] = float('inf')  # Exclude self
        least_similar_idx = np.argmin(similarities)
        
        print(f"  Most similar frame: {most_similar_idx} (similarity: {similarities[most_similar_idx]:.4f})")
        print(f"  Least similar frame: {least_similar_idx} (similarity: {similarities[least_similar_idx]:.4f})")


def analyze_logits(logits, tokenizer):
    """
    Analyze and display the model's output logits in text format
    """
    # Choose a random position in the sequence
    if not logits:
        print("No logits available to analyze.")
        return
        
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
    
    print(f"\nLogits analysis at step {logit_step}:")
    print(f"  Vocabulary size: {selected_logits.shape[0]}")
    print(f"  Min logit: {selected_logits.min().item():.4f}")
    print(f"  Max logit: {selected_logits.max().item():.4f}")
    
    print("\nTop 20 most likely next tokens:")
    for i in range(top_k):
        # Create a visual bar of probability
        bar_length = int(top_probs[i].item() * 50)
        bar = '#' * bar_length
        print(f"  {i+1}. '{top_tokens[i]}': {top_probs[i].item():.4f} {bar}")
    
    # Save full logits to file
    np.savetxt(f"logits_step_{logit_step}.txt", selected_logits.numpy())
    print(f"Full logits saved to logits_step_{logit_step}.txt")
    
    # Entropy of the distribution
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
    print(f"\nEntropy of distribution: {entropy:.4f} bits")
    if entropy < 2.0:
        print("  Low entropy - model is very confident in its prediction")
    elif entropy > 4.0:
        print("  High entropy - model is uncertain about prediction")
    else:
        print("  Medium entropy - model has moderate confidence")


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
    
    # Save frames before and after transformation
    save_sample_frames(original_frames, transformed_images)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(transformed_images.to('cuda'))

    print("\n===== FEATURE EXTRACTION RESULTS =====") 
    # Print feature information
    print("Encoder Image Features Shape:", features.shape)
    print("Encoder Image Features Sample (first 10 elements of first frame):", features[0, :10].tolist())
    
    # Analyze feature vector
    analyze_feature_vector(features)

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
    
    # Analyze logits
    print("\n===== MODEL LOGITS ANALYSIS =====")
    analyze_logits(logits, tokenizer)
    
    print("\nAll intermediate results have been demonstrated and saved to files.")
