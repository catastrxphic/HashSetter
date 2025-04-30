from flask import Flask, render_template, request
import img_detection
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('photo')
        if not file or file.filename == '':
            return "No file selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        prompt = request.form.get("prompt", "").strip()
        labels = img_detection.analyze_image_wlabels(filepath, user_prompt=prompt)
        hashtags = []

        for label in labels:
            fallback = img_detection.match_labels_to_hashtags([label])
            trending = img_detection.get_trending_hashtags_from_ritetag(label, fallback_hashtags=fallback)
            hashtags.extend(trending)

        prompt_keywords = img_detection.extract_prompt_keywords(prompt)
        for word in prompt_keywords:
            prompt_tags = img_detection.get_trending_hashtags_from_ritetag(word)
            hashtags.extend(prompt_tags)

        hashtags = list(dict.fromkeys(hashtags))

        # For the POST block
        return render_template('upload.html', hashtags=hashtags, labels=labels)

    # Default GET case
    return render_template('upload.html', hashtags=None, labels=None)



if __name__ == '__main__':
    app.run(debug=True)
