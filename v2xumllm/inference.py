import os
import sys
import random
import re
import cv2
import argparse
import torch
import pickle
from v2xumllm.constants import IMAGE_TOKEN_INDEX
from v2xumllm.conversation import conv_templates, SeparatorStyle
from v2xumllm.model.builder import load_pretrained_model
from v2xumllm.utils import disable_torch_init
from v2xumllm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import numpy as np
import clip
import faiss
from sentence_transformers import SentenceTransformer
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

# ============ FAISS & SBERT Setup ============

sbert_model = None
faiss_index = None
metadata_store = []

def initialize_sbert():
    global sbert_model, faiss_index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    faiss_index = faiss.IndexFlatL2(sbert_model.get_sentence_embedding_dimension())

def store_summary(text_summary, video_path):
    global faiss_index, metadata_store, sbert_model
    if sbert_model is None or faiss_index is None:
        initialize_sbert()

    embedding = sbert_model.encode(text_summary, convert_to_numpy=True, normalize_embeddings=True)
    faiss_index.add(np.array([embedding]))
    metadata_store.append({
        "text_summary": text_summary,
        "video_path": video_path
    })
    print(f"✅ Stored summary for: {video_path}")

def search_summaries(query_text, top_k=1):
    global faiss_index, metadata_store, sbert_model
    if sbert_model is None or faiss_index is None:
        initialize_sbert()

    query_embedding = sbert_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(np.array([query_embedding]), k=top_k)

    results = []
    for idx in I[0]:
        if idx < len(metadata_store):
            results.append(metadata_store[idx])
    return results

def save_faiss_and_metadata(index_path="faiss_index.index", metadata_path="metadata.pkl"):
    faiss.write_index(faiss_index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata_store, f)
    print("💾 FAISS index and metadata saved.")

def load_faiss_and_metadata(index_path="faiss_index.index", metadata_path="metadata.pkl"):
    global faiss_index, metadata_store, sbert_model
    initialize_sbert()

    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        print("📂 Loaded FAISS index.")

    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata_store = pickle.load(f)
        print("📂 Loaded metadata store.")

# ============ Helpers ============

def clean_output(text):
    return re.sub(r'\s*\[[^\]]*\]', '', text)

def extract_keyframes(text):
    return re.findall(r'\[([^\]]*)\]', text)

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

    input_token_len = input_ids.shape[1]
    decoded_outputs = tokenizer.batch_decode(outputs.sequences[:, input_token_len:], skip_special_tokens=True)[0]

    original_output = decoded_outputs.strip()
    if original_output.endswith(stop_str):
        original_output = original_output[:-len(stop_str)]
    original_output = original_output.strip()

    keyframes = extract_keyframes(original_output)
    cleaned_output = clean_output(original_output)
    
    return cleaned_output, keyframes, outputs.scores

# ============ Video Summarization Flow ============

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

    # Load previous FAISS index and metadata
    load_faiss_and_metadata()

    tokenizer, model, context_len = load_pretrained_model(args, args.stage2)
    model = model.cuda().to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval().cuda()

    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(num_frames / fps)
    cap.release()

    video_loader = VideoExtractor(N=duration)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    images = transform(images / 255.0).to(torch.float16)

    with torch.no_grad():
        features = clip_model.encode_image(images.cuda())

    query = "Please generate BOTH video and text summarization for this video."
    text_summary, keyframes, _ = inference(model, features, "<video>\n " + query, tokenizer)

    print("📝 Generated Text Summary:\n", text_summary)

    # Store this summary and save everything
    store_summary(text_summary, video_path)
    save_faiss_and_metadata()

    # Ask user if they want to query
    do_query = input("🔍 Do you want to query the summaries? (yes/no): ").strip().lower()
    if do_query == "yes":
        user_query = input("Enter your search query: ")
        results = search_summaries(user_query, top_k=3)

        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print("Text Summary:", result["text_summary"])
            print("Video Path:", result["video_path"])
    else:
        print("✅ Skipping search. Done.")
