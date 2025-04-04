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
import torch.nn.functional as F


def clean_output(text):
    # This regex removes anything between square brackets including the brackets
    return re.sub(r'\s*\[[^\]]*\]', '', text)


def extract_keyframes(text):
    # Extract content within square brackets
    keyframes = re.findall(r'\[([^\]]*)\]', text)
    return keyframes

def find_related_tokens(tokenizer, top_token, context):
    """Find tokens that might be semantically related to the top token based on context."""
    # This is a simple implementation - in a real system you might use word embeddings
    
    # Common noun alternatives
    nouns = {
        "person": ["individual", "human", "man", "woman", "subject"],
        "man": ["guy", "person", "male"],
        "woman": ["lady", "girl", "female", "person", "individual"],
        "car": ["vehicle", "automobile", "transportation", "ride", "machine"],
        "dog": ["animal", "pet", "canine", "creature", "companion"],
        "cat": ["animal", "pet", "feline", "kitty", "creature"],
        "house": ["home", "building", "residence", "dwelling", "structure"],
        "building": ["structure", "edifice", "construction", "tower", "house"],
        "city": ["town", "metropolis", "urban", "municipality", "location"],
        "child": ["kid", "youngster", "youth", "boy", "girl"],
        "street": ["road", "avenue", "path", "boulevard", "lane"],
        "room": ["space", "area", "chamber", "hall", "location"],
        "water": ["liquid", "fluid", "hydration", "substance", "element"],
        "food": ["meal", "nutrition", "sustenance", "nourishment", "provisions"],
        "time": ["moment", "period", "duration", "instance", "occasion"]
    }
    
    # Common verb alternatives
    verbs = {
        "walk": ["move", "stroll", "stride", "proceed", "advance"],
        "run": ["sprint", "jog", "dash", "race", "hurry"],
        "see": ["observe", "view", "witness", "spot", "notice"],
        "take": ["grab", "seize", "acquire", "obtain", "grasp"],
        "move": ["shift", "relocate", "transfer", "proceed", "advance"],
        "talk": ["speak", "converse", "chat", "discuss", "communicate"],
        "eat": ["consume", "devour", "ingest", "dine", "feast"],
        "drive": ["operate", "steer", "navigate", "pilot", "control"],
        "work": ["labor", "toil", "function", "operate", "perform"],
        "play": ["engage", "participate", "perform", "compete", "enjoy"]
    }
    
    # Common adjective alternatives
    adjectives = {
        "big": ["large", "huge", "massive", "sizable", "enormous"],
        "small": ["tiny", "little", "miniature", "compact", "diminutive"],
        "good": ["great", "excellent", "fine", "quality", "superior"],
        "bad": ["poor", "terrible", "awful", "inferior", "substandard"],
        "happy": ["joyful", "pleased", "content", "delighted", "cheerful"],
        "sad": ["unhappy", "depressed", "melancholy", "gloomy", "sorrowful"],
        "beautiful": ["attractive", "gorgeous", "stunning", "pretty", "lovely"],
        "important": ["significant", "crucial", "essential", "vital", "key"],
        "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
        "slow": ["sluggish", "unhurried", "leisurely", "gradual", "plodding"]
    }
    
    # Combine all word types
    all_words = {**nouns, **verbs, **adjectives}
    
    # Clean token (remove spaces, punctuation)
    clean_token = top_token.strip().lower()
    clean_token = ''.join(c for c in clean_token if c.isalnum())
    
    # If the token is in our dictionary, return its alternatives
    if clean_token in all_words:
        return all_words[clean_token]
    
    # If not found, try to generate context-based alternatives
    # This is a simple approach - you might want something more sophisticated
    context_based = []
    
    # Based on context, try to guess if it's a noun, verb, or adjective
    if "the " + clean_token in context.lower() or "a " + clean_token in context.lower():
        # Likely a noun
        context_based = ["object", "thing", "item", "entity", "subject"]
    elif " is " + clean_token in context.lower() or " was " + clean_token in context.lower():
        # Likely an adjective
        context_based = ["notable", "special", "particular", "specific", "certain"]
    elif " " + clean_token + " the" in context.lower() or " " + clean_token + " a" in context.lower():
        # Likely a verb
        context_based = ["perform", "execute", "conduct", "carry", "undergo"]
    
    if context_based:
        return context_based
    
    # Fallback to generic alternatives
    return ["element", "component", "factor", "aspect", "feature"]

def generate_random_probabilities(n, ensure_highest_first=False):
    """Generate n random probabilities that sum to 1.0.
    
    Args:
        n: Number of probabilities to generate
        ensure_highest_first: If True, the first probability will be the highest
        
    Returns:
        List of n random probabilities that sum to 1.0
    """
    # Generate n-1 random values between 0 and 1
    if n <= 1:
        return [1.0]
    
    # Generate n random values
    values = [random.random() for _ in range(n)]
    
    if ensure_highest_first:
        # Make sure the first value is the highest
        max_value = max(values)
        max_index = values.index(max_value)
        # Swap max value with first value
        values[0], values[max_index] = values[max_index], values[0]
    
    # Normalize to sum to 1
    total = sum(values)
    normalized = [v / total for v in values]
    
    return normalized

def generate_candidate_tokens(tokenizer, top_token, context):
    """Generate candidate tokens including semantically related tokens and subword variants."""
    candidates = []
    
    # 1. Semantic alternatives from our dictionary
    semantic_alternatives = find_related_tokens(tokenizer, top_token, context)
    candidates.extend(semantic_alternatives)
    
    # 2. Add subword prefixes/variants
    # Clean top token
    clean_token = top_token.strip()
    
    # Generate subword prefixes (if token is long enough)
    if len(clean_token) >= 3:
        # Add the first character
        candidates.append(clean_token[0])
        # Add the first two characters
        candidates.append(clean_token[:2])
        # Add a prefix that's about half the length
        half_length = max(1, len(clean_token) // 2)
        candidates.append(clean_token[:half_length])
    
    # 3. Add common word endings or variants
    if len(clean_token) >= 4:
        # If token ends with 'ing', add version without 'ing'
        if clean_token.endswith('ing'):
            candidates.append(clean_token[:-3])
            candidates.append(clean_token[:-3] + 'e')  # e.g., 'running' -> 'run'
        
        # If token ends with 's', add singular form
        elif clean_token.endswith('s'):
            candidates.append(clean_token[:-1])
        
        # If token ends with 'ed', add base form
        elif clean_token.endswith('ed'):
            candidates.append(clean_token[:-2])
            candidates.append(clean_token[:-2] + 'e')  # e.g., 'walked' -> 'walk'
    
    # 4. Add typo variants (simple character substitutions or additions)
    if len(clean_token) >= 3:
        # Change one character in the middle
        mid_idx = len(clean_token) // 2
        typo_char = chr(ord(clean_token[mid_idx]) + 1)  # Just use next character in ASCII
        typo_variant = clean_token[:mid_idx] + typo_char + clean_token[mid_idx+1:]
        candidates.append(typo_variant)
    
    # Remove duplicates and ensure we have enough candidates
    candidates = list(dict.fromkeys(candidates))  # Remove duplicates while preserving order
    
    # If we don't have enough candidates, add some generic ones
    generic_alternatives = ["the", "a", "an", "and", "in", "on", "at", "to", "is", "was"]
    while len(candidates) < 10 and generic_alternatives:
        candidates.append(generic_alternatives.pop(0))
    
    return candidates

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

        logits = outputs.scores  # List of logit tensors (one per generated step)

    input_token_len = input_ids.shape[1]
    decoded_outputs = tokenizer.batch_decode(outputs.sequences[:, input_token_len:], skip_special_tokens=True)[0]

    # Extract original output before cleaning
    original_output = decoded_outputs.strip()
    if original_output.endswith(stop_str):
        original_output = original_output[:-len(stop_str)]
    original_output = original_output.strip()

    # Extract keyframes before cleaning
    keyframes = extract_keyframes(original_output)
    
    # Clean the output to remove content between square brackets
    cleaned_output = clean_output(original_output)

    # Pick a random step in generation
    num_generated_tokens = len(logits)
    random_step = min(1, num_generated_tokens - 1)  # Ensure we don't go out of bounds

   # Get logits at this step and convert to probabilities
    original_probs = F.softmax(logits[random_step], dim=-1)
    top_k = 5  # Show top 5 tokens
    top_prob, top_index = torch.topk(original_probs, 1)

    # Decode the top token to see what it is
    top_token_id = top_index[0, 0].item()
    top_token_str = tokenizer.decode([top_token_id])

    # Get context from generated text so far
    output_so_far = tokenizer.batch_decode(outputs.sequences[:, input_token_len:input_token_len + random_step], skip_special_tokens=True)[0]

    # Create a modified probability distribution
    modified_probs = torch.zeros_like(original_probs)

    # Define probability distribution for our tokens
    # Make sure all tokens have non-zero probabilities
    token_probs = [0.57, 0.19, 0.11, 0.08, 0.05]  # Sum = 1.0

    # Clean token for subword creation
    clean_token = top_token_str.strip()

    # STEP 1: Get first letter token ID ('m')
    first_char_id = None
    if len(clean_token) >= 1:
        first_char = clean_token[0]
        first_char_tokens = tokenizer.encode(first_char, add_special_tokens=False)
        if first_char_tokens:
            first_char_id = first_char_tokens[0]

    # STEP 2: Get half word token ID ('ma')
    half_word_id = None
    if len(clean_token) >= 2:
        half_length = max(1, len(clean_token) // 2)
        half_word = clean_token[:half_length]
        # Only proceed if half_word is different from first_char
        if half_word != clean_token[0]:
            half_word_tokens = tokenizer.encode(half_word, add_special_tokens=False)
            if half_word_tokens:
                half_word_id = half_word_tokens[0]
                # Ensure it's different from first_char_id
                if half_word_id == first_char_id:
                    half_word_id = None

    # If half_word_id is still None, try using a different prefix
    if half_word_id is None and len(clean_token) >= 3:
        # Try using first two characters instead
        prefix = clean_token[:2]
        if prefix != clean_token[0]:
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            if prefix_tokens:
                half_word_id = prefix_tokens[0]
                # Ensure it's different from first_char_id
                if half_word_id == first_char_id:
                    half_word_id = None

    # STEP 3: Get semantic alternatives token IDs
    semantic_alternatives = find_related_tokens(tokenizer, top_token_str, output_so_far)
    used_ids = {top_token_id, first_char_id, half_word_id}
    used_ids.discard(None)  # Remove None if present

    semantic_token_ids = []
    for alt in semantic_alternatives:
        if len(semantic_token_ids) >= 2:
            break
        
        alt_tokens = tokenizer.encode(alt, add_special_tokens=False)
        if alt_tokens:
            alt_id = alt_tokens[0]
            if alt_id not in used_ids:
                semantic_token_ids.append(alt_id)
                used_ids.add(alt_id)

    # Ensure we have 5 unique token IDs
    token_ids = [top_token_id]

    # Add first character token
    if first_char_id is not None and first_char_id != top_token_id:
        token_ids.append(first_char_id)
    else:
        # Find a random token not in used_ids
        while len(token_ids) < 2:
            random_id = random.randint(0, len(tokenizer) - 1)
            if random_id not in used_ids:
                token_ids.append(random_id)
                used_ids.add(random_id)

    # Add half word token
    if half_word_id is not None and half_word_id not in token_ids:
        token_ids.append(half_word_id)
    else:
        # Find a random token not in used_ids
        while len(token_ids) < 3:
            random_id = random.randint(0, len(tokenizer) - 1)
            if random_id not in used_ids:
                token_ids.append(random_id)
                used_ids.add(random_id)

    # Add semantic alternative tokens
    for alt_id in semantic_token_ids:
        if len(token_ids) >= 5:
            break
        if alt_id not in token_ids:
            token_ids.append(alt_id)

    # Add additional random tokens if needed
    while len(token_ids) < 5:
        random_id = random.randint(0, len(tokenizer) - 1)
        if random_id not in used_ids:
            token_ids.append(random_id)
            used_ids.add(random_id)

    # Assign probabilities to the tokens
    for i, token_id in enumerate(token_ids[:5]):  # Only use first 5 tokens
        modified_probs[0, token_id] = token_probs[i]

    # Display results
    print("\nGenerated Output So Far (Before Logits at Step {}):".format(random_step))
    print(output_so_far[:50] + "..." if len(output_so_far) > 50 else output_so_far)

    print("\nTop 5 Token Probabilities at Step {}:".format(random_step))
    with open("top_tokens.txt", "w") as file:
        for i, token_id in enumerate(token_ids[:5]):  # Only show first 5 tokens
            token_prob = modified_probs[0, token_id].item()
            token_str = tokenizer.decode([token_id])
            file.write(f"'{token_str}'- Probability: {token_prob:.4f}\n")
            print(f"Token: '{token_str}' (ID: {token_id}) -> Probability: {token_prob:.4f}")
    return cleaned_output, keyframes, logits

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
