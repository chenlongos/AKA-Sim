import os
from flask import Blueprint, send_file

frontend_bp = Blueprint("frontend", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STATIC_DIR = os.path.join(ROOT_DIR, "static")
STATIC_3D_DIR = os.path.join(ROOT_DIR, "static-3d")

MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.ico': 'image/x-icon',
    '.json': 'application/json; charset=utf-8',
    '.wasm': 'application/wasm',
}

def _send_static(file_path):
    """发送静态文件，自动设置 MIME 类型"""
    if not os.path.isfile(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    mime = MIME_TYPES.get(ext)
    if mime:
        return send_file(file_path, mimetype=mime)
    return send_file(file_path)

@frontend_bp.route("/")
def serve_control():
    """控制台页面，由 frontend/control.html 构建而来"""
    index_html = os.path.join(STATIC_DIR, "control.html")
    result = _send_static(index_html)
    if result:
        return result
    return {"error": "Control page not built. Run: cd frontend && npm run build:control"}, 404

@frontend_bp.route("/assets/<path:path>")
def serve_control_assets(path):
    """控制台页面静态资源"""
    file_path = os.path.join(STATIC_DIR, "assets", path)
    result = _send_static(file_path)
    if result:
        return result
    return {"error": "Not Found"}, 404

@frontend_bp.route("/sim3d")
def serve_sim3d():
    """3D 模拟器页面，使用 Vite 构建的 3D 前端"""
    index_html = os.path.join(STATIC_3D_DIR, "index.html")
    result = _send_static(index_html)
    if result:
        return result
    return {"error": "3D frontend not built. Run: cd frontend && npm install && npm run build"}, 404

@frontend_bp.route("/sim3d/<path:path>")
def serve_sim3d_assets(path):
    """3D 前端静态资源"""
    file_path = os.path.join(STATIC_3D_DIR, path)
    result = _send_static(file_path)
    if result:
        return result
    return {"error": "Not Found"}, 404
