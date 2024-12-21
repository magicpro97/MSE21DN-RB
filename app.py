import io
import os
from concurrent.futures.thread import ThreadPoolExecutor
import cv2
import numpy as np
from flask import Flask, send_file, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from PIL import Image, ImageFilter
from rembg import remove
from rembg.session_factory import new_session

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
)
logger = app.logger
logger.setLevel('INFO')
pool = ThreadPoolExecutor(max_workers=4)


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
    logger.info('Processing frame in thread')
    global video_writer, background_image

    # Đọc và mở ảnh từ dữ liệu đầu vào
    frame_bytes = io.BytesIO(data['frame'])
    image = Image.open(frame_bytes).convert("RGBA")  # Chuyển sang RGBA ngay từ đầu để đảm bảo tính tương thích

    # Downscale hình ảnh để cải thiện hiệu suất
    original_size = image.size
    target_size = (512, 512)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Loại bỏ nền
    result = remove(image, session=session)

    # Khôi phục kích thước ban đầu
    final_image = result.resize(original_size, Image.Resampling.LANCZOS)

    # Làm mềm viên ảnh
    final_image = final_image.filter(ImageFilter.SMOOTH)

    # Chuyển ảnh thành mảng numpy cho OpenCV
    result_frame = np.array(final_image.convert("RGB"))

    # Tạo video_writer nếu chưa có
    if video_writer is None:
        height, width, _ = result_frame.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
    video_writer.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

    # Gửi lại khung hình đã xử lý cho front-end
    result_bytes = io.BytesIO()
    final_image.save(result_bytes, format="PNG")
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


if __name__ == '__main__':
    socketio.run(app, debug=True)
