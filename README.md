# 📸 HashSetter: Smart Hashtag Generator from Images
> *“The eye sees, the heart feels, and the algorithm deciphers.”*  

Upload a photo and prompt (optional) and wait for AI to analyze it and give you optimized, trending hashtags for Instagram — and more.

---

## 🧠 What It Does

**HashSetter** is an intelligent image analysis tool that intends to help creators, marketers, and casual users generate **context-aware**, **trend-sensitive** hashtags based on the content and items of their photos. 
With the help of CLIP (Contrastive Language–Image Pretraining), your photo is understood like a verse of poetry — and paired with the right aesthetic signals to reach kindred spirits online.


---

## ✨ Features

- ✅ Upload any image (JPG, PNG)
- 🧠 AI-powered image analysis with CLIP
- 🔍 Optional prompt to fine-tune context (e.g., "vintage aesthetic", "reading under sunlight")
- 📈 Suggests **relevant + trending** hashtags via integration with the **RiteTag API**
- 🌐 Instagram-focused (expandable to TikTok, X/Twitter, Pinterest)
- 💡 Intelligent fallback if trending data is unavailable
- ⚙️ Modular Python backend (easy to extend/customize)

---

## 🖼️ Example Use Case

> Uploading a photo of a book next to a cup of tea, with prompt: _“cozy morning”_  
AI detects: `book`, `tea`, `morning light`, `cozy aesthetic`  
Suggested Hashtags:  
> *“The eye sees, the heart feels, and the algorithm deciphers.”*  

`#bookstagram #cozymorning #tealovers #aestheticvibes #readersofig`

---

## 🔧 Tech Stack

| Component      | Tech Used                             |
|----------------|---------------------------------------|
| **🎨 Frontend**   | HTML, CSS                         |
| **🔮 Backend**    | Python, Flask                     |
| **🧠 AI Model**| OpenAI CLIP (Transformers)            |
| **🔗 Hashtag**| RiteTag (RiteKit)                      |
| **🧳 Hosting** | (To be added)                         |

---

## 🚀 How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/HashSetter.git
   cd HashSetter
2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
3. **Run the App**
   ```bash
   python app.py
4. Visit in browser
    ```bash
    Open http://localhost:5000

## To Do:
- [ ] UI improvements (cleaner, responsive design)
- [ ] Expand to TikTok and Twitter support
- [ ] Save hashtag bundles to user account
- [ ] Auto-caption suggestions based on prompt + image
- [ ] Light/dark mode toggle
- [ ] One-click copy button for hashtags

## 🤍 Why I Built This
As a lover of both aesthetics and algorithms, and a wannabe content creator ( [``@camis.locket``](https://www.instagram.com/camis.locket/) ), I wanted a tool that didn’t just process images, but understood them and helped like-minded people to find it -to create a community.
This is for the bookworms, the artists, the wanderers who craft each post like a diary entry, and want hashtags to match the soul of their story.
