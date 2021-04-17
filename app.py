import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

try:
    from PIL import Image
except ImportError:
    import Image

from inference import load_image_into_numpy_array, run_inference_for_single_image

LABEL_MAP_FILE = 'Your path to label_map.pbtxt'

tf.keras.backend.clear_session()

model = tf.saved_model.load('/your path to /Cust_od/exported_models/saved_model') 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('detection', filename=filename))
    return render_template('index.html')


@app.route('/detection/<filename>') 
def detection(filename):

    # Make prediction
    image_path = filename
    
    image_np = load_image_into_numpy_array(image_path)
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

    cv2.imshow(Image.fromarray(image_np))
    # new_img.save("Output.png","PNG")
    # return render_template('base.html')
if __name__ == '__main__':
    app.run(debug=True)

