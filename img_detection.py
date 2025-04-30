import streamlit as st
from PIL import Image
import torch
import clip
import os
from dotenv import load_dotenv
import requests
import re

load_dotenv()
RITEKIT_API_KEY = os.getenv("RITEKIT_API_KEY")

item_labels = [

    # TECH & PRODUCTIVITY
    "laptop", "computer keyboard", "mechanical keyboard", "tablet", "smartphone", "smartwatch",
    "notebook", "planner", "calendar", "journal", "desk", "clock", "watch", "glasses",
    "whiteboard", "lamp", "office chair", "backpack", "tote bag", "charger", "workspace",

    # STATIONERY & WRITING
    "pen", "highlighter", "sticky notes", "textbook", "flashcards", "calculator",
    "chalkboard", "exam sheet", "certificate", "syllabus", "paper", "open book", "self-help book",

    # CAFE & FOOD AESTHETIC
    "coffee cup", "latte", "matcha", "espresso machine", "croissant", "cake", "pastry",
    "dessert plate", "tea set", "teapot", "sugar cube", "saucer", "menu", "barista station", "mug",

    # CAFE SPACE & VIBES
    "cafe table", "cafe chair", "receipt", "window seat", "café lighting", "plant pot",

    # ART & CREATIVITY
    "paint palette", "paintbrush", "colored pencil", "sketchbook", "charcoal", "ink", "markers",
    "canvas", "art easel", "art studio", "pencil", "collage", "typography", "artwork", "journal spread",

    # MUSIC & SOUND
    "piano keyboard", "digital piano", "acoustic guitar", "vinyl record", "record player",
    "headphones", "microphone", "studio monitor", "music sheet", "audio mixer",

    # HOME & COZY
    "bed", "blanket", "pillow", "rug", "curtain", "sofa", "window", "mirror", "wall art",
    "bookshelf", "fairy lights", "string lights", "workspace", "kitchen table",

    # BEAUTY & SELF-CARE
    "mirror selfie", "skincare set", "lotion bottle", "face mask", "robe", "bath", "bathtub",
    "towel", "essential oil", "crystal", "incense", "perfume bottle", "makeup brush", "lipstick",
    "nail polish", "candle", "candlelight", "hairbrush", "meditation cushion",

    # FASHION
    "hat", "dress", "trench coat", "scarf", "earrings", "necklace", "rings", "jewelry box",
    "closet", "outfit grid", "vintage shirt", "denim jacket", "heels", "boots", "umbrella",
    "sunglasses", "nail art",

    # TRAVEL
    "passport", "suitcase", "luggage", "airplane", "airport", "train station", "bus",
    "street sign", "skyline", "map", "hotel room", "hotel key", "balcony", "rooftop",
    "bridge", "tower", "monument", "museum", "travel journal", "travel cup", "hiking boots",

    # NATURE & LANDSCAPE
    "beach", "ocean", "waves", "mountain trail", "forest path", "lake", "waterfall", "flower",
    "sunrise", "sunset", "clouds", "sky", "sand", "tree", "leaf", "field", "wildflowers",
    "tent", "bonfire", "cabin", "campfire", "boat", "rock", "shell", "nature trail", "grass",

    # EVENTS & CELEBRATIONS
    "birthday cake", "balloons", "gift box", "party hat", "confetti", "champagne", "holiday lights",
    "fireworks", "lantern", "costume", "mask", "parade", "traditional clothing", "flag",
    "crowd", "feast", "decoration",

    # PETS & ANIMALS
    "cat", "dog", "rabbit", "horse", "fish", "bird", "leash", "pet bed", "paw print", "collar",
    "fur brush", "pet food", "birdcage",

    # SYMBOLIC / EMOTIVE ELEMENTS
    "sunlight", "shadow", "reflection", "silhouette", "fog", "moon", "night sky", "stars", "galaxy",
    "rain", "snow", "wind", "smoke", "dust", "puddle", "motion blur", "backlight", "overexposure",
    "flicker",

    # VISUAL MEDIA / CONTENT CREATION
    "film camera", "digital camera", "photo frame", "photo print", "polaroid", "editing screen",

    # VISION & IDEATION
    "quote board", "vision board", "chalk writing", "mood board", "handwritten letter", "typewriter",
    "journal entry"
]



hashtag_map = {
    "laptop": ["#laptoplife", "#digitalworkspace", "#studygram"],
    "coffee": ["#coffeetime", "#coffeevibes", "#caféculture"],
    "books": ["#bookstagram", "#readingnook", "#literarylife"],
    "mountain": ["#mountainview", "#hikingadventures", "#natureescape"],
    "beach": ["#beachvibes", "#oceanbreeze", "#sunandsand"],
    "desk": ["#deskinspo", "#workspacegoals", "#aestheticdesk"],
    "plant": ["#plantlover", "#plantdecor", "#greenvibes"],
    "cake": ["#cakelover", "#desserttime", "#sweettooth"],
    "mirror": ["#mirrorpic", "#mirrorvibes", "#selfreflection"],
    "sunset": ["#sunsetlovers", "#goldenhour", "#skyfire"],
    "cat": ["#catsofinstagram", "#meowdel", "#caturday"],
    "dog": ["#dogsofinstagram", "#puppylove", "#furryfriend"],
    "journal": ["#journalingcommunity", "#aestheticjournal", "#stationeryaddict"],
    "forest": ["#forestwalk", "#greennature", "#intothewoods"],
    "camera": ["#filmphotography", "#shotonfilm", "#capturethemoment"],
    "painting": ["#artistsoninstagram", "#paintingvibes", "#creativeexpression"],
    "skincare": ["#skincareroutine", "#glowup", "#selfcare"],
    "bed": ["#cozybed", "#sundayvibes", "#sleepyaesthetic"],
    "candle": ["#candlevibes", "#cozynight", "#homedecor"],
    "notebook": ["#notesonnotes", "#studyvibes", "#bulletjournal"],
    "art": ["#artlovers", "#creativeflow", "#dailyart"],
    "keyboard": ["#mechanicalkeyboard", "#keyboardlife", "#techsetup"],
    "hat": ["#hatstyle", "#fashionfit", "#hatlover"],
    "sunglasses": ["#sunnies", "#sunstyle", "#shadeson"]
}

# adding the model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# tokenize labels 
text_inputs = torch.cat([clip.tokenize(label) for label in item_labels]).to(device)

"""
    Step 1: Analyze image, identify items & get labels

    This function will be taking the image, encoding & normalizing the features
    and computing similarity agains the words (to see which items match which words). 
    It will then return the top 5 labels to generate hashtags based on those terms - on another function

    Update: On top of the AI detection, an optional user prompt was added to be analyzed with the label and 
    increase accuracy of the system
"""

def analyze_image_wlabels(image_path, user_prompt=None, top_k=5):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenize the main item label list
    text_inputs = torch.cat([clip.tokenize(label) for label in item_labels]).to(device)

    # If user prompt provided, tokenize it and append it to the list
    if user_prompt:
        prompt_input = clip.tokenize(user_prompt).to(device)
        text_inputs = torch.cat([text_inputs, prompt_input])

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # Normalize both
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_indices = similarity[0].topk(top_k)

        # If prompt was included, exclude it from the label results
        index_offset = 1 if user_prompt else 0
        label_matches = [item_labels[i] for i in top_indices if i < len(item_labels)]

    return label_matches

"""
    Step 1.5: Get User Prompt's Keywords:

    This function will take the most important words of the prompt to be used to curate better hashtags
"""

def extract_prompt_keywords(prompt, top_n=3):
    if not prompt:
        return []
    
    # Remove punctuation, lowercase, and split
    words = re.findall(r'\b\w+\b', prompt.lower())

    # Optionally: Filter stopwords (basic list)
    stopwords = {'with', 'the', 'and', 'at', 'my', 'in', 'on', 'a', 'of', 'for', 'to', 'an'}
    keywords = [word for word in words if word not in stopwords]

    # Return the top N unique keywords (or all)
    return list(dict.fromkeys(keywords))[:top_n]
    
"""
    Step 2: Matching Top Labels to Hashtags

    This function will use previous output as input and map to its matching hashtags

    Update: instead of using the hashmap, I will be using a connection to RiteTag to get trending hashtags
"""

def match_labels_to_hashtags(labels):
    hashtags = []

    for label in labels:
        hashtags.extend(hashtag_map.get(label, []))

    return hashtags

def get_trending_hashtags_from_ritetag(keyword, fallback_hashtags=None, max_tags=5):
    if not RITEKIT_API_KEY:
        print("⚠️ Missing RiteTag API key.")
        return fallback_hashtags or []

    url = f"https://api.ritekit.com/v1/stats/hashtag-suggestions?text={keyword}&client_id={RITEKIT_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        hashtag_entries = data.get("data", [])  # <-- this was the key you missed

        if not hashtag_entries:
            print(f"No trending hashtags found for: {keyword}")
            return fallback_hashtags or []

        hashtags = [f"#{entry['tag']}" for entry in hashtag_entries[:max_tags]]
        print(f"✅ Trending for '{keyword}': {hashtags}")
        return hashtags

    except Exception as e:
        print(f"Error getting trending hashtags for '{keyword}': {e}")
        return fallback_hashtags or []
