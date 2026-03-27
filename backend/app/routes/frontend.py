import os
from flask import Blueprint, render_template, send_file

frontend_bp = Blueprint("frontend", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STATIC_DIR = os.path.join(ROOT_DIR, "static")
STATIC_3D_DIR = os.path.join(ROOT_DIR, "static-3d")

@frontend_bp.route("/sim3d")
def serve_sim3d():
    """3D 模拟器页面，使用 Vite 构建的 3D 前端"""
    index_html = os.path.join(STATIC_3D_DIR, "index.html")
    if os.path.exists(index_html):
        return send_file(index_html)
    return {"error": "3D frontend not built. Run: cd frontend && npm install && npm run build"}, 404

@frontend_bp.route("/sim3d/<path:path>")
def serve_sim3d_assets(path):
    """3D 前端静态资源"""
    file_path = os.path.join(STATIC_3D_DIR, path)
    if os.path.isfile(file_path):
        return send_file(file_path)
    return {"error": "Not Found"}, 404

@frontend_bp.route("/", defaults={"path": ""})
@frontend_bp.route("/<path:path>")
def serve_react(path):
    if path.startswith("api"):
        return {"error": "Not Found"}, 404
    return render_template("index.html")
