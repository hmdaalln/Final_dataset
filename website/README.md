# DnCNN Image Denoiser

A Flask project with the same layout as the cat/dog demo: app.py, templates, static, model, requirements.txt and Dockerfile. Includes your original dncnn_best.pth, selected at epoch 43 (validation MSE 0.00038356507334023394). No retraining or ONNX conversion is required for this version.

## Run on your Mac

1. Unzip this project. Move the `dncnn-denoiser` folder to your Desktop.
2. Use Python 3.11 or 3.12 (recommended). Open Terminal and run:

```bash
cd ~/Desktop/dncnn-denoiser
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

3. Open **http://127.0.0.1:5000** in Chrome. Keep Terminal running.
4. Choose a noisy image, click **Denoise image**, compare the two images, and download the PNG.
5. Press Control+C in Terminal to stop the app.

The first dependency installation can take several minutes. Later, you only need to activate `.venv` and run `python app.py`. If `python3` is an unsupported version, install Python 3.11/3.12 and use `python3.11`/`python3.12` for the venv command. Intel Macs may require a torch version supported by their macOS and Python version; use the Docker route if no matching wheel is available.

If port 5000 is busy (sometimes used by AirPlay Receiver), run:

```bash
python -m flask --app app run --host 127.0.0.1 --port 5001
```

Then open http://127.0.0.1:5001. Do not double-click index.html; the Python backend must run.

## Files

| File | Purpose |
| --- | --- |
| app.py | Model architecture, checkpoint loading, upload validation, and image denoising |
| templates/index.html | Upload and before/after page |
| static/style.css | Page styling |
| static/app.js | Send image to Python and display/download the response |
| static/uploads/ | Reserved folder; images are not permanently saved here |
| model/dncnn_best.pth | Your original trained checkpoint |
| requirements.txt | Python dependencies |
| Dockerfile | Optional container setup |
| tests/test_app.py | Backend and numerical tests |

## How inference works

The architecture exactly matches the notebook: 17 convolution layers, 128 features, RGB, BatchNorm and ReLU. The checkpoint is loaded strictly with `weights_only=True`; never replace it with an untrusted file. The app runs on CPU for portability.

Images are checked by their actual contents, rotated according to EXIF, converted to RGB (transparency on white), and resized to at most 512 × 512 while preserving aspect ratio. Both displayed images use that same resized input. The model receives float32 values in [0,1].

**Denoised output = noisy image − predicted residual**, clipped to [0,1] and saved as PNG. To keep memory manageable, inference uses 96-pixel tiles with 17 pixels of context on each side. This covers the model's receptive field; it is not an extra image filter. Downloads use the processed size, not necessarily the original resolution.

The app accepts up to 10 MB and 25 megapixels per image. It processes one request at a time. For a presentation, start with a held-out noisy image from your test set; results on unrelated noise may differ. No PSNR or SSIM is shown without a matching reference image. Showing a pleasing result alone is not a quantitative evaluation.

## Optional Docker

Install and start Docker Desktop, then inside the project folder run:

```bash
docker build -t dncnn-denoiser .
docker run --rm -p 127.0.0.1:5000:5000 dncnn-denoiser
```

Open http://127.0.0.1:5000. Docker uses a CPU-only PyTorch install and Gunicorn. The container configuration is included but was not built in this environment.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover a real checkpoint inference request, image errors, missing uploads, transparency, resize behavior, and tiled versus full-image inference. They do not replace browser UI testing.

## Sharing and privacy

This is a **local project**, not a public link. localhost works only on the computer running the app. You can demonstrate it by sharing your screen. Public deployment would need a Python-capable host, HTTPS, access controls, rate limits and operational review; do not expose Flask's development server directly.

Images are sent to the Python server (your own computer for local use); the app does not keep them in project folders. The model is outside the static directory and is not served as a download. The ZIP includes the original checkpoint including optimizer state, so share the ZIP only with people you intend to give those weights to.

Flask references: [Uploads](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/) and [development server](https://flask.palletsprojects.com/en/stable/api/).
