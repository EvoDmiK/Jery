"""Small authenticated HTTP adapter for the n8n v1 contract."""
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from jery.data.collector import encode


def make_server(collector, token, host='127.0.0.1', port=8080):
    if not isinstance(token, str) or len(token) < 16 or not token.isascii() or token.strip() != token:
        raise ValueError('Provide an ASCII bearer token of at least 16 characters')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Requests and errors must not print potentially sensitive client data.

        def respond(self, status, body):
            body = encode(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            self.connection.settimeout(10)
            provided = self.headers.get('Authorization', '')
            if not hmac.compare_digest(provided.encode(), ('Bearer ' + token).encode()):
                self.respond(401, {'error': 'unauthorized'})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            parsed = urlsplit(self.path)
            if parsed.path != '/v1/routing-candidates':
                self.respond(404, {'error': 'not_found'})
                return
            try:
                query = parse_qs(parsed.query, keep_blank_values=True)
                if set(query) - {'consumer', 'limit'} or any(len(v) != 1 for v in query.values()):
                    raise ValueError()
                batch = collector.fetch(query.get('consumer', ['jery-dataset-v1'])[0], int(query.get('limit', ['100'])[0]))
                self.respond(200, batch)
            except ValueError:
                self.respond(400, {'error': 'invalid_request'})
            except Exception:
                self.respond(500, {'error': 'collection_failed'})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/review-batches':
                self.respond(404, {'error': 'not_found'})
                return
            try:
                if self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding'):
                    raise ValueError()
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError()
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError()
                self.respond(200, collector.commit(json.loads(raw)))
            except (ValueError, UnicodeError):
                self.respond(400, {'error': 'invalid_batch'})
            except Exception:
                self.respond(500, {'error': 'persistence_failed'})

    return ThreadingHTTPServer((host, port), Handler)
