from  flask import Flask, request, jsonify, send_file
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import io
import os

app = Flask(__name__)

@app.route('/remove-background/image', methods=['POST'])
def remove_background_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file)
    result = remove(image)

    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    return send_file(output, mimetype='image/png')

# API để xử lý video
@app.route('/remove-background/video', methods=['POST'])
def remove_background_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    temp_input = "input.mp4"
    temp_output = "output.mp4"
    file.save(temp_input)

    cap = cv2.VideoCapture(temp_input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        result = remove(image)
        result_frame = np.array(result)
        out.write(result_frame)

    cap.release()
    out.release()
    os.remove(temp_input)
    return send_file(temp_output, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
