"""Serve the built site beneath the GitHub Pages prefix for integration tests."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=4173)
args = parser.parse_args()


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/path-planning-ode/"):
            self.path = self.path[len("/path-planning-ode") :]
            try:
                super().do_GET()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when tests deliberately abort runtime downloads.
        else:
            self.send_error(404)


directory = Path(__file__).resolve().parents[1] / "web" / "dist"
ThreadingHTTPServer(
    ("127.0.0.1", args.port), partial(Handler, directory=str(directory))
).serve_forever()
