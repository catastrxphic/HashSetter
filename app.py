from flask import Flask, render_template, request
import img_detection
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('photo')
        if not file or file.filename == '':
            return "No file selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 🔍 Analyze image and get top items
        labels = img_detection.analyze_image_wlabels(filepath)
        hashtags = img_detection.match_labels_to_hashtags(labels)

        return render_template('upload.html', hashtags=hashtags, labels=labels)

    return render_template('upload.html', hashtags=None, labels=None)

if __name__ == '__main__':
    app.run(debug=True)
