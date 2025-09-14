import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['COLORIZED_FOLDER'] = 'colorized_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload and colorized folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COLORIZED_FOLDER'], exist_ok=True)

# Allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the colorization model
#create model folder store bellow three lines files
prototxt_path = 'model/colorization_deploy_v2.prototxt' # https://github.com/richzhang/colorization/tree/caffe/colorization/models
model_path = 'model/colorization_release_v2.caffemodel' # https://github.com/dath1s/colorizor/blob/main/colorization_release_v2.caffemodel
kernel_path = 'model/pts_in_hull.npy' # https://github.com/dath1s/colorizor/blob/main/pts_in_hull.npy

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path)

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, dtype="float32")]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize_image():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_filepath)

        # Colorization logic
        image = cv2.imread(original_filepath)
        if image is None:
            return render_template('index.html', error='Could not read image file.')

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = (255.0 * colorized).astype("uint8")

        colorized_filename = "colorized_" + filename
        colorized_filepath = os.path.join(app.config['COLORIZED_FOLDER'], colorized_filename)
        cv2.imwrite(colorized_filepath, colorized)

        return render_template('index.html', original_image=filename, colorized_image=colorized_filename)
    else:
        return render_template('index.html', error='Invalid file type')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/colorized_images/<filename>')
def colorized_file(filename):
    return send_from_directory(app.config['COLORIZED_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['COLORIZED_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)