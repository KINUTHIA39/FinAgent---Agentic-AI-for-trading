"""Serve the FinGPT vs FinBERT dashboard locally.

    python scripts/serve_fingpt.py          # http://localhost:8050
    python scripts/serve_fingpt.py 8080     # pick a port

Serves results/fingpt/dashboard. The page fetches compare.json, so opening
index.html straight from the filesystem will not work (browsers block fetch on
file:// URLs). Rebuild the data with scripts/export_fingpt_compare.py.
"""
import functools
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "results" / "fingpt" / "dashboard"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8050


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT),
                                functools.partial(Handler, directory=str(DIR))) as s:
        url = f"http://localhost:{PORT}"
        print(f"FinGPT vs FinBERT dashboard: {url}   (Ctrl+C to stop)")
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        s.serve_forever()
