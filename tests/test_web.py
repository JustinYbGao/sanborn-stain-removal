import http.client
import io
import threading
import unittest
from email.parser import BytesParser
from email.policy import default

import numpy as np
from PIL import Image

from sanborn_stain.web import make_server
from tests.test_paper import drawing


class StainRemovalWebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=30)
        try:
            connection.request(method, path, body)
            response = connection.getresponse()
            return response.status, response.getheader('Content-Type'), response.read()
        finally:
            connection.close()

    def test_stained_tiff_returns_only_before_after_and_native_rgb_png(self):
        status, _, page = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'Remove stains', page)
        clean, x, y = drawing()
        spot = np.exp(-((x-86)**2+(y-104)**2)/(2*21**2))
        raw = np.rint(clean - spot[..., None]*[15, 23, 40]).astype(np.uint8)
        upload = io.BytesIO()
        Image.fromarray(raw).save(upload, format='TIFF')

        status, content_type, body = self.request('POST', '/remove-stains', upload.getvalue())
        self.assertEqual(status, 200, body[:300])
        message = BytesParser(policy=default).parsebytes(
            f'Content-Type: {content_type}\r\n\r\n'.encode() + body)
        parts = {p.get_param('name', header='Content-Disposition'): p.get_payload(decode=True)
                 for p in message.iter_parts()}
        self.assertEqual(set(parts), {'original', 'preview', 'cleaned'})
        with Image.open(io.BytesIO(parts['cleaned'])) as result:
            self.assertEqual((result.format, result.mode, result.size), ('PNG', 'RGB', (360, 260)))
            corrected = np.array(result)
        self.assertGreater(int(corrected[104, 86, 2]), int(raw[104, 86, 2]))
        np.testing.assert_array_equal(corrected[42:103, 163:250], raw[42:103, 163:250])
        for name, expected in (('original', raw), ('preview', corrected)):
            with Image.open(io.BytesIO(parts[name])) as preview:
                np.testing.assert_array_equal(np.asarray(preview), expected)

    def test_rejects_non_image_upload(self):
        status, _, body = self.request('POST', '/remove-stains', b'Not an image')
        self.assertEqual(status, 400)
        self.assertIn(b'image', body)


if __name__ == '__main__':
    unittest.main()
