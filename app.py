from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os, shutil

#pip list --format=freeze

app = Flask(__name__)


MODEL_PATH = 'models/model_1.h5'
model = load_model(MODEL_PATH)

def predict(img_file, model):

    img = image.load_img(img_file, target_size=(192,192))
    img_arr = np.array(img)
    img_arr_dims = np.expand_dims(img_arr, axis=0)
    img_nor = img_arr_dims/255

    pred = model.predict(img_nor) # [[0.00027]]
    if pred[0][0] > 0.75:
        return "Glucoma Not Detected"
    else:
        return "Glucoma Detected"

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

UPLOAD_FOLDER = os.getcwd() + '/tmp/'

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename('temp'+f.filename[-4:]))
        f.save(file_path)
        # Make prediction
        preds = predict(file_path, model)

        return render_template('result.html', data=preds)
    return None

if __name__ == '__main__':
    app.run(debug=True)
