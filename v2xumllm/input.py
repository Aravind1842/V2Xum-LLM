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
    
    # Get video information
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS of video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    duration = int(num_frames / fps)  # Duration in seconds
    
    # Display video information
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {num_frames}")
    print(f"Duration: {duration} seconds")
    
    # Extract frames using your VideoExtractor
    video_loader = VideoExtractor(N=duration)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})
    
    # Select 3 random frames
    if len(images) >= 3:
        random_indices = random.sample(range(len(images)), 3)
        
        # Create output directory
        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save frames in tensor format
        for i, idx in enumerate(random_indices):
            frame = images[idx]
            output_path = os.path.join(output_dir, f"frame_{idx}.jpg")
            
            try:
                frame_np = frame.cpu().numpy()
                if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # Convert CHW to HWC
                cv2.imwrite(output_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                print(f"Saved frame {idx}")
            except Exception as e:
                print(f"Error saving frame {idx}: {e}")
                print(f"Frame type: {type(frame)}, Shape: {frame.shape if hasattr(frame, 'shape') else 'Unknown'}")
    else:
        print(f"Not enough frames extracted. Only got {len(images)} frames.")
    
    print("Done!")
