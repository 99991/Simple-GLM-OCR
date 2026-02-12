from simpleglmocr import SimpleGlmOcr
from PIL import Image
import io
import http.server
import socketserver
import email
import email.policy

# Initialize the model once
print("Loading model...")
model = SimpleGlmOcr()

class OCRRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/process' or self.path == '/api/ocr':
            # Get content type and length
            content_type = self.headers.get('Content-Type')
            content_length = int(self.headers.get('Content-Length', 0))

            if not content_type or not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Content-Type must be multipart/form-data")
                return

            body = self.rfile.read(content_length)

            # Prepend the Content-Type header so the parser knows the boundary
            msg_data = f"Content-Type: {content_type}\r\n".encode('ascii') + body
            msg = email.message_from_bytes(msg_data, policy=email.policy.default)

            prompt = None
            image_data = None

            for part in msg.iter_parts():
                name = part.get_param('name', header='content-disposition')
                if name == 'prompt':
                    prompt = part.get_content().strip()
                elif name == 'image':
                    image_data = part.get_payload(decode=True)

            if not prompt or not image_data:
                self.send_error(400, "Missing prompt or image")
                return

            # Read image data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            print(f"Processing request with prompt: {prompt}")
            
            # Run model
            text = model.run(prompt, image)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(text.encode('utf-8'))
        else:
            self.send_error(404)

def run_server(port=8000):
    handler = OCRRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Server started at http://127.0.0.1:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
