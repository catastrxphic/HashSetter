import streamlit as st
from PIL import Image
import torch
import clip
import os
from dotenv import load_dotenv
import requests

load_dotenv()
RITEKIT_API_KEY = os.getenv("RITEKIT_API_KEY")

item_labels = [
        "laptop", "keyboard", "tablet", "phone", "notebook", "planner", "calendar", "journal", "sticky notes", 
        "books", "pen", "desk", "clock", "watch", "glasses", "whiteboard", "lamp", "office chair", "backpack", 
        "tote bag", "mug", "candle", "headphones", "charger", "shelf", "coffee", 
        "tea", "espresso machine", "reading light",
        "coffee cup", "latte", "croissant", "cake", "dessert", "teapot", "matcha", "pastry", "cafe table", 
        "cafe chair", "coffee beans", "sugar cube", "saucer", "menu", "barista", "receipt", "window seat", 
        "café light", "plant pot",
        "beach", "ocean", "waves", "mountain", "forest", "lake", "waterfall", "trail", "flower", "sky", 
        "clouds", "sunrise", "sunset", "sand", "rock", "shell", "leaf", "tree", "grass", "wildflowers", 
        "field", "nature path", "tent", "bonfire", "boat", "campfire", "cabin",
        "passport", "suitcase", "luggage", "airplane", "airport", "train station", "bus", "street sign", 
        "skyline", "building", "map", "hotel room", "hotel key", "balcony", "rooftop", "bridge", "tower", 
        "museum", "monument", "travel journal", "camera", "guidebook", "hiking boots", "travel cup",
        "mirror", "skincare", "lotion", "face mask", "bath", "bathtub", "towel", "robe", 
        "essential oil", "crystal", "incense", "meditation cushion", "plant", "flowers", "perfume", 
        "candlelight", "hairbrush", "nail polish", "sunlight", "self-help book",
        "dumbbells", "resistance band", "yoga mat", "treadmill", "water bottle", "sports shoes", 
        "activewear", "fitness tracker", "gym bag", "protein shake", "gym mirror", "locker", "basketball", 
        "tennis racket", "bike", "running track",
        "paint", "paintbrush", "palette", "canvas", "sketchbook", "pencil", "colored pencil", "ink", 
        "charcoal", "markers", "art studio", "easel", "sculpture", "film camera", "digital camera", 
        "photo frame", "collage", "journal spread", "editing screen",
        "textbook", "highlighter", "flashcards", "calculator", "chalkboard", 
        "classroom", "graduation cap", "diploma", "school bag", "lecture hall", "syllabus", "exam sheet", 
        "certificate",
        "quote board", "vision board", "open book", "paper", "handwritten letter", "photo print", 
        "polaroid", "mood board", "typography", "artwork", "journal entry", "chalk writing", "shadow", 
        "silhouette", "window light", "raindrops",
        "hat", "dress", "sunglasses", "scarf", "boots", "heels", "earrings", "necklace", "rings", 
        "mirror selfie", "closet", "jewelry box", "makeup brush", "lipstick", "perfume bottle", 
        "outfit grid", "nail art", "denim jacket", "trench coat", "umbrella", "vintage shirt",
        "bed", "blanket", "pillow", "window", "curtain", "bookshelf", "sofa", "rug",  "wall art", 
        "record player", "vinyls", "mug", "fairy lights", "string lights", "workspace", 
        "kitchen table", "notebook stack",
        "birthday cake", "balloons", "gift box", "party hat", "confetti", "champagne", 
        "holiday lights", "fireworks", "lantern", "costume", "mask", "traditional clothing", "flag", 
        "parade", "crowd", "feast", "decoration",
        "cat", "dog", "bird", "fish", "horse", "rabbit", "pet bed", "leash", "collar", "paw print", 
        "fur brush", "pet food", "birdcage",
        "rain", "snow", "fog", "wind", "flame", "smoke", "dust", "moon", "stars", "galaxy", "night sky", 
        "shadows", "reflection", "puddle", "backlight", "flicker", "overexposure", "blur", "motion"
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
"""

def analyze_image_wlabels(image_path, top_k = 5):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # normalization process
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1,keepdim = True)

        # similarity computation
        sim = (100.0* image_features @ text_features.T).softmax(dim=-1)
        top_perc, top_lab = sim[0].topk(top_k)

        return [item_labels[idx] for idx in top_lab]
    
"""
    Setp 2: Matching Top Labels to Hashtags

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
