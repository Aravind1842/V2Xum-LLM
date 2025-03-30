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
            output_scores=True,
            return_dict_in_generate=True
        )

        logits = outputs.scores

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


def create_keyframe_video(video_path, keyframe_segments, output_path, duration_per_frame):
    # Open the original video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Flatten and unique the keyframe segments
    all_keyframes = sorted(set([frame for segment in keyframe_segments for frame in segment]))
    
    # Write keyframes to the output video
    for frame_idx in all_keyframes:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Write the frame multiple times to create a longer duration
            for _ in range(int(fps * duration_per_frame)):
                out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Summarized video saved to {output_path}")
    return output_path


def display_video_info(video_path):
    """Display comprehensive information about the input video."""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = num_frames / fps
    
    # Get video codec
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(fourcc & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)
    
    # Print video information
    print("\n" + "="*50)
    print("VIDEO INFORMATION:")
    print("="*50)
    print(f"Path: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds ({int(duration//60)}m {int(duration%60)}s)")
    print(f"Total Frames: {num_frames}")
    print(f"Codec: {codec}")
    print(f"File Size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print("="*50)
    
    cap.release()
    return width, height, fps, num_frames, duration


def display_random_frames(video_path, num_frames=3):
    """Extract and display random frames from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select random frames
    random_indices = sorted(random.sample(range(total_frames), min(num_frames, total_frames)))
    
    plt.figure(figsize=(15, 5*num_frames))
    
    for i, frame_idx in enumerate(random_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB for correct display in matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            plt.subplot(num_frames, 1, i+1)
            plt.imshow(frame_rgb)
            plt.title(f"Frame {frame_idx} of {total_frames}")
            plt.axis('off')
    
    cap.release()
    plt.tight_layout()
    plt.savefig("random_frames.png")
    print("\nRandom frames saved to random_frames.png")
    plt.close()
    
    return random_indices


def display_transformation_comparison(video_path, random_indices, transform):
    """Show comparison of frames before and after CLIP transformation."""
    cap = cv2.VideoCapture(video_path)
    
    plt.figure(figsize=(15, 10))
    
    for i, frame_idx in enumerate(random_indices[:2]):  # Limit to 2 frames for clarity
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Original frame (convert BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame for CLIP
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
            transformed_tensor = transform(frame_tensor)
            
            # Convert transformed tensor back to numpy for display
            # Unnormalize
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
            transformed_display = transformed_tensor.clone().cpu()
            for t, m, s in zip(transformed_display, mean, std):
                t.mul_(s).add_(m)
            
            transformed_np = transformed_display.permute(1, 2, 0).numpy()
            transformed_np = np.clip(transformed_np, 0, 1)
            
            # Display original frame
            plt.subplot(2, 2, i*2+1)
            plt.imshow(frame_rgb)
            plt.title(f"Original Frame {frame_idx}")
            plt.axis('off')
            
            # Display transformed frame
            plt.subplot(2, 2, i*2+2)
            plt.imshow(transformed_np)
            plt.title(f"Transformed Frame {frame_idx} (224x224)")
            plt.axis('off')
    
    cap.release()
    plt.tight_layout()
    plt.savefig("transformation_comparison.png")
    print("\nTransformation comparison saved to transformation_comparison.png")
    plt.close()


def display_features_info(features):
    """Display information about the features extracted by CLIP."""
    print("\n" + "="*50)
    print("CLIP FEATURES INFORMATION:")
    print("="*50)
    print(f"Features Shape: {features.shape}")
    print(f"Features Type: {features.dtype}")
    print(f"Features Range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"Features Mean: {features.mean().item():.4f}")
    print(f"Features Std: {features.std().item():.4f}")
    
    # Visualize feature distribution for the first frame
    if features.shape[0] > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(features[0].cpu().numpy(), bins=50)
        plt.title("Distribution of CLIP Features (First Frame)")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.savefig("feature_distribution.png")
        print("\nFeature distribution saved to feature_distribution.png")
        plt.close()
        
        # Visualize feature heatmap for the first few dimensions
        plt.figure(figsize=(12, 8))
        plt.imshow(features[:min(10, features.shape[0]), :50].cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar(label='Feature Value')
        plt.title("CLIP Features Heatmap (First 10 frames x 50 dimensions)")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Frame Index")
        plt.savefig("feature_heatmap.png")
        print("\nFeature heatmap saved to feature_heatmap.png")
        plt.close()
    
    print("="*50)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Keyframe Summarization Demo")
    parser.add_argument("--clip_path", type=str, default="/content/V2Xum-LLM-Models/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="/content/V2Xum-LLM-Models/llava-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="/content/V2Xum-LLM-Models/v2xumllm-vicuna-v1-5-7b-stage2-e2")
    parser.add_argument("--video_path", type=str, default="demo/Ex1.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2)
    model = model.cuda()
    model.to(torch.float16)

    # 1) Display input video information
    video_path = args.video_path
    width, height, fps, num_frames, duration = display_video_info(video_path)
    
    # 2) Display random frames from the input video
    random_frame_indices = display_random_frames(video_path, num_frames=3)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=int(duration))
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 3) Show comparison of frames before and after transformation
    display_transformation_comparison(video_path, random_frame_indices, transform)

    # Transform images
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))
    
    # 4) Display information about extracted features
    display_features_info(features)

    prompts = {
        "V-sum": ["Please generate a VIDEO summarization for this video."],
        "T-sum": ["Please generate a TEXT summarization for this video."],
        "VT-sum": ["Please generate BOTH video and text summarization for this video."]
    }

    query = random.choice(prompts["VT-sum"])
    text_summary, keyframes, _ = inference(model, features, "<video>\n " + query, tokenizer)

    print("\nText Summary:", text_summary)
    
    print("\nKeyframes Identified:")
    keyframe_segments = []
    if keyframes:
        for i, keyframe_str in enumerate(keyframes):
            keyframe_nums = [int(k) for k in keyframe_str.split(",")]
            scaled_keyframes = [int((k / 100) * num_frames) for k in keyframe_nums]
            unique_scaled_keyframes = sorted(list(set(scaled_keyframes)))
            
            print(f"Segment {i+1}: {', '.join(map(str, unique_scaled_keyframes))}")
            keyframe_segments.append(unique_scaled_keyframes)
    else:
        print("No keyframes were identified in the output.")
        
    # Create summarized video
    if keyframe_segments:
        output_video_path = create_keyframe_video(
            video_path, 
            keyframe_segments, 
            output_path="video_summary.mp4", 
            duration_per_frame=1
        )
        print(f"\nSummarized video saved to: {output_video_path}")
