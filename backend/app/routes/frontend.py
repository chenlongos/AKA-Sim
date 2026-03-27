import os
from flask import Blueprint, render_template, send_file

frontend_bp = Blueprint("frontend", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@frontend_bp.route("/sim3d")
def serve_sim3d():
    """3D 模拟器页面，使用独立 main.html（Three.js + TensorFlow.js）"""
    main_html = os.path.join(ROOT_DIR, "main.html")
    if os.path.exists(main_html):
        return send_file(main_html)
    return {"error": "main.html not found"}, 404

@frontend_bp.route("/", defaults={"path": ""})
@frontend_bp.route("/<path:path>")
def serve_react(path):
    if path.startswith("api"):
        return {"error": "Not Found"}, 404
    return render_template("index.html")
