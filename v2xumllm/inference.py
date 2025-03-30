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

    video_loader = VideoExtractor(N=10)  # Adjust N as needed
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    # Select 3 random frames
    if len(images) >= 3:
        random_indices = random.sample(range(len(images)), 3)
        output_dir = "output"
        transformed_dir = "transformed_frames"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(transformed_dir, exist_ok=True)

        transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        for i, idx in enumerate(random_indices):
            frame = images[idx]
            output_path = os.path.join(output_dir, f"frame_{idx}.png")
            transformed_path = os.path.join(transformed_dir, f"frame_{idx}.png")

            try:
                frame_np = frame.cpu().numpy()
                if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                    frame_np = np.transpose(frame_np, (1, 2, 0))  # Convert CHW to HWC
                cv2.imwrite(output_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

                # Apply transformation
                frame_tensor = torch.tensor(frame_np).permute(2, 0, 1) / 255.0  # Convert to tensor and normalize
                transformed_frame = transform(frame_tensor).permute(1, 2, 0).numpy()  # Convert back to numpy
                transformed_frame = (transformed_frame * 255).astype(np.uint8)  # Denormalize
                cv2.imwrite(transformed_path, cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
    else:
        print(f"Not enough frames extracted. Only got {len(images)} frames.")

    print("Done!")
