"""Tiny reverse proxy: rewrites thinking.enabled → thinking.adaptive for muyuan.do.

Usage:  py thinking-proxy.py
Then set ANTHROPIC_BASE_URL to http://127.0.0.1:8787 in ccswitch.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import http.client
import json
import ssl
import sys

TARGET_HOST = "muyuan.do"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8787


class Proxy(BaseHTTPRequestHandler):
    def _forward(self, method):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        rewritten = False
        if body:
            try:
                data = json.loads(body)
                thinking = data.get("thinking")
                if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                    data["thinking"] = {"type": "adaptive"}
                    data["output_config"] = {"effort": "max"}
                    body = json.dumps(data).encode()
                    rewritten = True
                    print("[proxy]  OK thinking.enabled -> thinking.adaptive")
            except Exception:
                pass

        conn = http.client.HTTPSConnection(
            TARGET_HOST,
            context=ssl._create_unverified_context(),
            timeout=120,
        )
        fwd_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        try:
            conn.request(method, self.path, body=body, headers=fwd_headers)
            resp = conn.getresponse()
        except Exception as e:
            self.send_error(502, f"Upstream error: {e}")
            conn.close()
            return

        self.send_response(resp.status)
        skip = {"transfer-encoding", "connection", "keep-alive"}
        for k, v in resp.getheaders():
            if k.lower() not in skip:
                self.send_header(k, v)
        self.end_headers()

        try:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")

    def do_OPTIONS(self):
        self._forward("OPTIONS")

    def do_DELETE(self):
        self._forward("DELETE")

    def do_PUT(self):
        self._forward("PUT")

    def do_PATCH(self):
        self._forward("PATCH")

    def log_message(self, fmt, *args):
        print(f"[proxy] {fmt % args}")


if __name__ == "__main__":
    print(f"Thinking proxy: http://{LISTEN_HOST}:{LISTEN_PORT} -> https://{TARGET_HOST}")
    print(f"  Rewrites thinking.enabled -> thinking.adaptive + output_config.effort")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        HTTPServer((LISTEN_HOST, LISTEN_PORT), Proxy).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)