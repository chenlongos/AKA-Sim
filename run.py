import os
import threading
import signal
import time

from backend.app import create_app
from backend.app.extensions import socketio

app = create_app()

def run_http():
    default_port = 5000 if os.name == "nt" else 80
    port = int(os.getenv("APP_HTTP_PORT", str(default_port)))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)

def run_https():
    cert_path = os.getenv("APP_CERT_PATH", "/root/AKA-00/cert.pem")
    key_path = os.getenv("APP_KEY_PATH", "/root/AKA-00/key.pem")
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        return
    default_port = 5443 if os.name == "nt" else 443
    port = int(os.getenv("APP_HTTPS_PORT", str(default_port)))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True, ssl_context=(cert_path, key_path))

shutdown_event = threading.Event()

def signal_handler(_sig, _frame):
    print("Shutting down ...")
    shutdown_event.set()
    os._exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    http_thread = threading.Thread(target=run_http, daemon=True)
    https_thread = threading.Thread(target=run_https, daemon=True)

    http_thread.start()
    https_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        signal_handler(None, None)
