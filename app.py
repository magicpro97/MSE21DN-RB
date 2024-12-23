import io
import os
from concurrent.futures.thread import ThreadPoolExecutor
import cv2
import numpy as np
from flask import Flask, send_file, send_from_directory, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from PIL import Image, ImageFilter
from rembg import remove
from rembg.session_factory import new_session
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
)
logger = app.logger
logger.setLevel('INFO')
pool = ThreadPoolExecutor(max_workers=4)

# Khởi tạo Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None
session = new_session(model_name="u2netp")

# Default page route
@app.route('/')
def index():
    return render_template('web.html')

@socketio.on('reset')
def reset_video():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

def process_frame(data):
    logger.info('Processing frame with Mediapipe (RGBA)')
    global video_writer

    # Read and open the image from the input data in RGBA format
    frame_bytes = io.BytesIO(data['frame'])
    image = Image.open(frame_bytes).convert("RGBA")  # Convert to RGBA

    # Convert image to NumPy array
    frame = np.array(image)

    # Mediapipe requires frames to be in RGB format (BGR for OpenCV processing)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert from RGBA to RGB

    # Predict the mask using Mediapipe
    result = selfie_segmentation.process(frame_rgb)
    mask = result.segmentation_mask

    # Threshold to separate foreground and background
    threshold = 0.5
    mask_binary = (mask > threshold).astype(np.uint8) * 255  # Convert to binary

    # Prepare the RGBA frame
    frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)

    # Set the alpha channel using the binary mask
    frame_rgba[:, :, 3] = mask_binary

    # Optional: Replace the background with transparency
    transparent_background = np.zeros_like(frame_rgba)
    transparent_background[:, :, 3] = 255 - mask_binary  # Invert mask for transparency

    # Combine the foreground with the transparent background
    result_frame = frame_rgba

    # Initialize video_writer if not already created
    if video_writer is None:
        height, width, _ = result_frame.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    # Write the RGBA frame to the video
    video_writer.write(cv2.cvtColor(result_frame, cv2.COLOR_RGBA2BGRA))  # OpenCV uses BGRA format for video

    # Send the processed frame back to the front-end
    result_image = Image.fromarray(result_frame, mode='RGBA')
    result_bytes = io.BytesIO()
    result_image.save(result_bytes, format="PNG")
    socketio.emit('processed_frame', result_bytes.getvalue())

@socketio.on('frame')
def handle_frame(data):
    logger.info('Received frame')
    pool.submit(process_frame, data)


@app.route('/download-video', methods=['GET'])
def download_video():
    global video_writer
    if video_writer:
        video_writer.release()
        video_writer = None
    # Check if the file exists before sending
    if os.path.exists(output_video_path):
        return send_file(output_video_path, as_attachment=True)
    else:
        return jsonify({'error': 'Video file not found'}), 404


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

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    socketio.run(app, debug=True)
