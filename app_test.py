import subprocess
import time

from flask import Flask, request, send_file, jsonify, url_for
from PIL import Image, ImageFilter
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route('/transfer/v1/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Xử lý ảnh
        image = Image.open(filepath)
        processed_image = image.filter(ImageFilter.BLUR)
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        processed_image.save(processed_filepath)

        # Trả về URL của ảnh đã xử lý
        processed_url = url_for('static', filename=f'processed_images/{processed_filename}', _external=True)
        return jsonify({"processed_image_url": processed_url}), 200


def run_ngrok(port):
    process = subprocess.Popen(f'ngrok tcp {port} --log "stdout"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        elif b'url=' in output:
            output = output.decode()
            output = output[output.index('url=tcp://') + 10: -1]
            return output.split(':')


if __name__ == '__main__':
    port = 5000
    ngrok_url = run_ngrok(port)
    print(f" * Running on {ngrok_url}")
    app.run(debug=True,host='0.0.0.0', port=5000)
