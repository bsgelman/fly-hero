"""Serve Fly Hero locally with browser caching turned off, so edits always show on reload.

Run: python tools/serve.py   then open http://localhost:8766
"""
import functools
import http.server
import pathlib


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


root = pathlib.Path(__file__).resolve().parent.parent
http.server.test(HandlerClass=functools.partial(NoCache, directory=str(root)), port=8766)
