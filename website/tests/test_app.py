import base64
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import app as demo


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.client = demo.app.test_client()

    def image_file(self, size=(40, 32), mode='RGB'):
        image = Image.new(mode, size, (100, 140, 180, 0) if mode == 'RGBA' else (100, 140, 180))
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        stream.seek(0)
        return stream

    def test_home_and_static(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Epoch 43', response.data)
        for filename in ['style.css', 'app.js']:
            self.assertEqual(self.client.get('/static/' + filename).status_code, 200)
        self.assertEqual(self.client.get('/model/dncnn_best.pth').status_code, 404)

    def test_missing_and_invalid_upload(self):
        self.assertEqual(self.client.post('/denoise').status_code, 400)
        response = self.client.post('/denoise', data={'image': (io.BytesIO(b'not an image'), 'fake.png')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_oversize(self):
        response = self.client.post('/denoise', data={'image': (io.BytesIO(b'x' * (12 * 1024 * 1024)), 'large.png')})
        self.assertEqual(response.status_code, 413)

    def test_resize_and_transparency(self):
        image, old_size = demo.prepare_image(self.image_file((800, 400), 'RGBA'))
        self.assertEqual(old_size, (800, 400))
        self.assertEqual(image.size, (512, 256))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_real_inference_request(self):
        response = self.client.post('/denoise', data={'image': (self.image_file(), '../../input.png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['epoch'], 43)
        raw = base64.b64decode(response.json['result'].split(',')[1])
        result = Image.open(io.BytesIO(raw))
        self.assertEqual(result.size, (40, 32))
        self.assertEqual(result.mode, 'RGB')
        expected = demo.denoise_image(Image.open(self.image_file()).convert('RGB'))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        self.assertFalse(np.array_equal(np.asarray(result), np.asarray(Image.open(self.image_file()))))

    def test_tiles_match_full_inference(self):
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, (137, 151, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        tiled = np.asarray(demo.denoise_image(image)).astype(np.int16)
        x = torch.from_numpy((pixels.astype(np.float32) / 255).transpose(2, 0, 1).copy()).unsqueeze(0)
        with torch.inference_mode():
            full = (x - demo.model(x)).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
        full = np.rint(full * 255).astype(np.int16)
        self.assertLessEqual(np.abs(tiled - full).max(), 1)

    def test_busy(self):
        demo.inference_lock.acquire()
        try:
            response = self.client.post('/denoise', data={'image': (self.image_file(), 'input.png')})
            self.assertEqual(response.status_code, 429)
        finally:
            demo.inference_lock.release()

    def test_error_releases_lock(self):
        with patch.object(demo, 'denoise_image', side_effect=RuntimeError('test failure')):
            response = self.client.post('/denoise', data={'image': (self.image_file(), 'input.png')})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(demo.inference_lock.locked())


if __name__ == '__main__':
    unittest.main()
