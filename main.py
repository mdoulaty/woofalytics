import logging
import json

import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import glob
from record import Woofalytics

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Main")


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("./html/main.html", "r") as f:
                self.wfile.write(f.read().encode())
        elif self.path == "/rec":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("./html/record.html", "r") as f:
                self.wfile.write(f.read().encode())
        elif self.path.startswith("/store-record"):
            query_params = parse_qs(urlparse(self.path).query)
            button_number = query_params.get("button", [None])[0]
            if button_number and button_number == "rec":
                wa.store_clip()
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                wav_files_count = len(list(glob.glob("*.wav")))
                self.wfile.write(
                    f"Thanks for your contributions. This will help Teddy and other furry friends!\n There are {wav_files_count} recordings in storage.".encode()
                )
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path.startswith("/api/bark"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            json_data = wa.get_last_pred().copy()
            # bark_probability is a list, pick the max?
            if len(json_data["bark_probability"]) > 1:
                json_data["bark_probability"] = max(json_data["bark_probability"])
            else:
                json_data["bark_probability"] = 0

            self.wfile.write(json.dumps(json_data).encode())
        else:
            self.send_response(404)
            self.end_headers()


def term_handler(signum, frame):
    logger.info("Ctrl+C pressed.")
    wa.stop()
    exit(1)


signal.signal(signal.SIGINT, term_handler)
wa = Woofalytics()


def run_server(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


def main():
    logger.info("Starting Woofalytics server, press Ctrl+C to stop...")
    wa.start()
    run_server()


if __name__ == "__main__":
    main()
