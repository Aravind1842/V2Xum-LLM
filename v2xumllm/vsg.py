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
    
    return output_path


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
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Transform images
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    
    # Extract features
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))

    prompts = {
        "V-sum": ["Please generate a VIDEO summarization for this video."],
        "T-sum": ["Please generate a TEXT summarization for this video."],
        "VT-sum": ["Please generate BOTH video and text summarization for this video."]
    }

    query = random.choice(prompts["VT-sum"])
    text_summary, keyframes, _ = inference(model, features, "<video>\n " + query, tokenizer)
    keyframe_segments = []
    if keyframes:
        for i, keyframe_str in enumerate(keyframes):
            keyframe_nums = [int(k) for k in keyframe_str.split(",")]
            scaled_keyframes = [int((k / 100) * duration) for k in keyframe_nums]
            unique_scaled_keyframes = sorted(list(set(scaled_keyframes)))
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
        
    vocab_file = "text_sum.txt"
    
    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write(text_summary)
    
    print(f"Text Summary saved to {vocab_file}")
