"""Run the local web app or correct one original scan."""

import argparse
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError

from .files import restore_file
from .web import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, help='Original 8-bit RGB TIFF, JP2, PNG, or JPEG')
    parser.add_argument('output_directory', nargs='?', type=Path, help='New folder for the cleaned PNG and diagnostics')
    parser.add_argument('--serve', action='store_true', help='Open the local web server')
    parser.add_argument('--port', default=8792, type=int, help='Local web port (default: 8792)')
    args = parser.parse_args()
    if args.serve:
        if args.source or args.output_directory:
            parser.error('Use --serve without input and output paths.')
        if not 0 <= args.port <= 65535:
            parser.error('Port must be between 0 and 65535.')
        with make_server(args.port) as server:
            print(f'Sanborn Stain Removal: http://127.0.0.1:{server.server_port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.source is None or args.output_directory is None:
        parser.error('Provide SOURCE and NEW_OUTPUT_DIRECTORY, or use --serve.')
    cv2.setNumThreads(2)
    try:
        print(restore_file(args.source, args.output_directory))
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        parser.exit(1, f'Could not clean the image: {error}\n')


if __name__ == '__main__':
    main()
