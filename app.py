import os

from flask import Flask, send_file, request, jsonify
from flask_socketio import SocketIO, emit
from rembg import remove
from PIL import Image, ImageOps
import cv2
import numpy as np
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Kích hoạt WebSocket

background_image = None
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None


@socketio.on('frame')
def handle_frame(data):
    global video_writer

    frame_bytes = io.BytesIO(data['frame'])
    image = Image.open(frame_bytes)

    # Xử lý xóa nền
    result = remove(image)

    # Áp dụng background nếu có
    global background_image
    if background_image:
        background = ImageOps.fit(background_image, result.size, method=Image.Resampling.LANCZOS)
        final_image = Image.alpha_composite(background, result)
    else:
        final_image = result

    result_frame = np.array(final_image.convert("RGB"))

    if video_writer is None:
        height, width, _ = result_frame.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
    video_writer.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

    # Gửi lại khung hình đã xử lý cho front-end
    result_bytes = io.BytesIO()
    final_image.save(result_bytes, format="PNG")
    emit('processed_frame', result_bytes.getvalue())


@app.route('/upload-background', methods=['POST'])
def upload_background():
    global background_image
    file = request.files['background']
    background_image = Image.open(file).convert("RGBA")
    return "Background uploaded successfully", 200


@app.route('/download-video', methods=['GET'])
def download_video():
    global video_writer
    if video_writer:
        video_writer.release()
        video_writer = None
    return send_file(output_video_path, as_attachment=True)


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


@app.route('/remove-background/video', methods=['POST'])
def remove_background_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    temp_input = "input.mp4"
    temp_output = "output.mp4"
    file.save(temp_input)

    cap = cv2.VideoCapture(temp_input)
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
    socketio.run(app, debug=True)
