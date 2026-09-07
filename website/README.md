# Image Denoiser

This project is an image denoising web application. It uses our trained DnCNN model to reduce noise from an uploaded image.

The model was trained from scratch using residual learning. It predicts the noise in the image, then subtracts it from the noisy input.

**Denoised output = Noisy image − Predicted residual**

## Live Website

The website is available here:

[Open Image Denoiser](https://image-denoiser-dwo0.onrender.com)

The website is hosted on Render using the free plan. The first visit may take around one minute to load if the website has been inactive.

## How to Use the Website

1. Open the website.
2. Click **Choose an image**.
3. Select a JPG or PNG image.
4. Click **Denoise image**.
5. Wait for the model to process the image.
6. Compare the original and denoised images.
7. Click **Download PNG** to save the result.

## Model Information

The application uses our best DnCNN checkpoint from epoch 43.

The model has:

* 17 convolutional layers
* 128 features
* RGB image support
* Batch Normalization
* ReLU activation
* Residual learning

The best validation MSE was approximately **0.000384**.

## Project Files

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `app.py`               | Loads the model and processes uploaded images |
| `templates/index.html` | Contains the website structure                |
| `static/style.css`     | Contains the website design                   |
| `static/app.js`        | Uploads the image and displays the result     |
| `model/dncnn_best.pth` | Contains the trained model checkpoint         |
| `requirements.txt`     | Lists the required Python libraries           |
| `Dockerfile`           | Used to deploy the website on Render          |
| `tests/test_app.py`    | Contains tests for the application            |

## How the Application Works

The uploaded image is checked and converted to RGB. Large images are resized to a maximum of 512 pixels while keeping the same aspect ratio.

The image is converted to values between 0 and 1 and passed to the trained model. The model predicts the noise residual, which is subtracted from the noisy image.

The final image is limited to values between 0 and 1 and converted to PNG format.

The uploaded images are processed in memory and are not permanently saved inside the project folders.

## Run the Project Locally

To run the website on a Mac, open Terminal inside the project folder and use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

Keep Terminal open while using the local website. Press `Control + C` to stop it.

## Notes

* The application accepts JPG and PNG images.
* The maximum upload size is 50 MB.
* Processing may take longer on the free Render plan.
* Results may be different depending on the image and type of noise.
* PSNR and SSIM are not calculated on the website because they require a matching reference image.

