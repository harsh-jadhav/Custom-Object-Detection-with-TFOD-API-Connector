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
        import numpy as np
        import cv2

        # Using OpenCV to initialize the webcam
        cap = cv2.VideoCapture(0) # put 0 for webcam input or 'filename.mp4' for video file in disk

        while cap.isOpened():
            ret, image_np = cap.read()
            
            detections = run_inference_for_single_image(model, image_np)

            label_id_offset = 1
            
            image_np_with_detections = image_np.copy()

            vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  detections['detection_boxes'],
                  (detections['detection_classes']+ label_id_offset).astype(int),
                  detections['detection_scores'],
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=.50,
                  line_thickness=4,
                  agnostic_mode=False)
            
        #     display(Image.fromarray(image_np_with_detections))
            cv2.imshow('connector', image_np_with_detections)
            if cv2.waitKey(1) == 13: #13 is the Enter Key
                break
                
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()  




if __name__ == '__main__':
    app.run(debug=True)

