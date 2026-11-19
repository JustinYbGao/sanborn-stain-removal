"""Serve local stain removal for original Sanborn scans."""

import argparse
import io
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

import cv2
from PIL import Image, UnidentifiedImageError

from .spots import restore_rgb
from .paper import read_rgb8

HERE = Path(__file__).resolve().parent
MAX_UPLOAD = 256 * 1024 * 1024
MAX_PIXELS = 80_000_000
PREVIEW_SIZE = (1800, 1800)


def preview_png(image):
    preview = image.copy() if max(image.size) <= PREVIEW_SIZE[0] else image.resize(
        tuple(max(1, round(side * PREVIEW_SIZE[0] / max(image.size))) for side in image.size),
        Image.Resampling.LANCZOS,
    )
    buffer = io.BytesIO()
    preview.save(buffer, format='PNG')
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, media_type, data):
        self.send_response(status)
        self.send_header('Content-Type', media_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def error(self, status, message):
        self.reply(status, 'application/json', json.dumps({'error': message}).encode())

    def do_GET(self):
        assets = {
            '/': ('index.html', 'text/html; charset=utf-8'),
            '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
            '/style.css': ('style.css', 'text/css; charset=utf-8'),
        }
        if self.path not in assets:
            self.error(404, 'Page not found.')
            return
        name, media_type = assets[self.path]
        self.reply(200, media_type, (HERE / name).read_bytes())

    def do_POST(self):
        if self.path != '/remove-stains':
            self.error(404, 'Page not found.')
            return
        try:
            size = int(self.headers.get('Content-Length', ''))
        except ValueError:
            size = 0
        if size <= 0:
            self.error(400, 'Choose an image to upload.')
            return
        if size > MAX_UPLOAD:
            self.error(413, 'Choose an image smaller than 256 MiB.')
            return
        if not self.server.processing.acquire(blocking=False):
            self.error(409, 'Another image is being processed. Please try again shortly.')
            return
        try:
            self.remove_stains(size)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.processing.release()

    def remove_stains(self, size):
        with TemporaryDirectory(prefix='sanborn-stain-') as folder:
            source = Path(folder) / 'upload'
            target = Path(folder) / 'cleaned.png'
            try:
                with source.open('wb') as stream:
                    while size:
                        block = self.rfile.read(min(size, 1024 * 1024))
                        if not block:
                            raise ValueError('The image upload was incomplete.')
                        stream.write(block)
                        size -= len(block)
                with Image.open(source) as header:
                    if header.width * header.height > MAX_PIXELS:
                        self.error(413, 'Choose an image with at most 80 million pixels.')
                        return
                raw, _ = read_rgb8(source)
                cleaned, _, _, _ = restore_rgb(raw)
                original_preview = preview_png(Image.fromarray(raw))
                result = Image.fromarray(cleaned)
                cleaned_preview = preview_png(result)
                result.save(target, format='PNG')
            except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
                self.error(400, 'Could not read the image. Use an 8-bit RGB TIFF, JP2, PNG, or JPEG.')
                return
            except ValueError as err:
                self.error(400, str(err))
                return
            self.send_result(original_preview, cleaned_preview, target)

    def send_result(self, original, preview, target):
        boundary = uuid4().hex
        names = ('original', 'preview', 'cleaned')
        headers = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.png"\r\n'
            'Content-Type: image/png\r\n\r\n'
            for name in names
        ]
        headers = [header.encode('ascii') for header in headers]
        end = f'--{boundary}--\r\n'.encode('ascii')
        length = sum(map(len, headers)) + len(original) + len(preview) + target.stat().st_size + len(end) + 6
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        for header, data in zip(headers[:2], (original, preview)):
            self.wfile.write(header)
            self.wfile.write(data)
            self.wfile.write(b'\r\n')
        self.wfile.write(headers[2])
        with target.open('rb') as stream:
            shutil.copyfileobj(stream, self.wfile)
        self.wfile.write(b'\r\n' + end)


def make_server(port):
    cv2.setNumThreads(2)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.processing = Lock()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=8792, type=int)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f'Sanborn Stain Removal: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
