# ============================================================
# AKA-Sim Dockerfile
# Multi-stage: dev (development) + prod (production)
# ============================================================

# ---------- Base ----------
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci

# ---------- Development ----------
FROM base AS dev

# Install dev tools
RUN pip install --no-cache-dir \
    debugpy \
    watchdog \
    flake8 \
    black

# Keep all source mounted, expose ports
# Port 80: Flask HTTP
# Port 5000: Flask fallback HTTP (Windows)
# Port 5173: Vite dev server
EXPOSE 80 5000 5173

CMD ["python", "run.py"]

# ---------- Production ----------
FROM base AS prod

# Build frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Copy backend source
COPY backend/ ./backend/
COPY run.py .
COPY https_init.sh .
COPY init.sh .
COPY main.html .
COPY templates/ ./templates/

# Copy built frontend to static folder
RUN cp -r frontend/dist/. static/ 2>/dev/null || mkdir -p static && cp -r frontend/dist/. static/

EXPOSE 80 443

CMD ["sh", "-c", "chmod +x /app/https_init.sh && /app/https_init.sh && python run.py"]
