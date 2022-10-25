from flask import Flask, request, jsonify, url_for, render_template
import os
import uuid
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.preprocessing import image

from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO

ALLOWED_EXTENSION = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSION


app = Flask(__name__)
model = MobileNet(weights='imagenet', include_top=True)


@app.route('/')
def index():
    return render_template('ImageML.html')


@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No image uploaded')
    file = request.files['image']

    if file.filename == '':
        return render_template('ImageML.html', prediction="You didn't select an image")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # filename = file.filename
        print("---" + filename + "---")
        x_inp = []
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

        x_inp = image.img_to_array(img)
        x_inp = np.expand_dims(x_inp, axis=0)
        x_inp = preprocess_input(x_inp)
        pred = model.predict(x_inp)
        # return the 3 most likely predictions in this image
        predictions_as_list = decode_predictions(pred, top=3)

        items = []
        for item in predictions_as_list[0]:
            items.append({'name': item[1], 'prob': float(item[2])})

        response = {'pred': items}
        return render_template('ImageML.html', prediction=f'Image is most likely {response}')

    else:
        return render_template('ImageML.html', prediction='Invalid file extension')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
