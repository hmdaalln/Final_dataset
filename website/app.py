"""Local DnCNN demo: upload -> predict residual -> display and download."""
import base64
import io
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 51 * 1024 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_SIDE = 512
Image.MAX_IMAGE_PIXELS = 25_000_000
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'dncnn_best.pth'
# CPU works on Macs and other laptops without needing CUDA.
torch.set_num_threads(min(4, os.cpu_count() or 1))
inference_lock = threading.Lock()


class DnCNN(nn.Module):
    """Same 17-layer, 128-feature RGB architecture as the notebook."""
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers.extend([
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(128, 3, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, noisy):
        return self.model(noisy)


# Load the existing best checkpoint. This does not train the model.
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
model = DnCNN()
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
MODEL_EPOCH = int(checkpoint['epoch'])
VALIDATION_LOSS = float(checkpoint['validation_loss'])
del checkpoint


def prepare_image(stream):
    """Validate image contents, fix rotation, and limit the demo size."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(stream) as opened:
            if opened.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('Please upload a JPG, PNG or WebP image.')
            if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError('Image is too large. Please use an image under 25 megapixels.')
            opened.load()
            corrected = ImageOps.exif_transpose(opened)
            rgba = corrected.convert('RGBA')
    rgb = Image.new('RGB', rgba.size, 'white')
    rgb.paste(rgba, mask=rgba.getchannel('A'))
    original_size = rgb.size
    rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return rgb, original_size


@torch.inference_mode()
def denoise_image(image):
    """Use overlapping context to reduce memory without tile seams."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    output = torch.empty_like(noisy)
    height, width = array.shape[:2]
    core, halo = 96, 17  # 17 convolutions give a 17-pixel receptive radius.
    for y in range(0, height, core):
        for x in range(0, width, core):
            top, left = max(0, y - halo), max(0, x - halo)
            bottom = min(height, y + core + halo)
            right = min(width, x + core + halo)
            tile = noisy[:, :, top:bottom, left:right]
            clean_tile = tile - model(tile)
            end_y, end_x = min(height, y + core), min(width, x + core)
            output[:, :, y:end_y, x:end_x] = clean_tile[
                :, :, y - top:end_y - top, x - left:end_x - left
            ]
    pixels = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(pixels * 255).astype(np.uint8))


def png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


@app.get('/')
def home():
    return render_template('index.html', epoch=MODEL_EPOCH, val_loss=VALIDATION_LOSS)


@app.post('/denoise')
def denoise():
    uploaded = request.files.get('image')
    if uploaded is None or not uploaded.filename:
        return jsonify(error='Please choose an image first.'), 400
    data = uploaded.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        return jsonify(error='Please use an image smaller than 10 MB.'), 413
    try:
        image, original_size = prepare_image(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        return jsonify(error='Cannot read this image. Use a valid JPG, PNG or WebP under 25 megapixels.'), 400
    # One inference at a time prevents several requests exhausting memory.
    if not inference_lock.acquire(blocking=False):
        return jsonify(error='The model is busy. Please try again shortly.'), 429
    try:
        start = time.perf_counter()
        result = denoise_image(image)
        seconds = round(time.perf_counter() - start, 2)
        return jsonify(original=png_data_url(image), result=png_data_url(result),
                       width=image.width, height=image.height,
                       original_width=original_size[0], original_height=original_size[1],
                       seconds=seconds, epoch=MODEL_EPOCH)
    except Exception:
        app.logger.exception('Denoising failed')
        return jsonify(error='Processing failed. Please try a smaller image.'), 500
    finally:
        inference_lock.release()


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Upload too large. Please choose an image smaller than 10 MB.'), 413


@app.after_request
def protect_response(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response


if __name__ == '__main__':
    print(f'DnCNN checkpoint loaded: epoch {MODEL_EPOCH}')
    app.run(host='127.0.0.1', port=5000, debug=False)
