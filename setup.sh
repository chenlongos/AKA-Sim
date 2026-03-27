#!/bin/bash

set -e

IMAGE_NAME="docker.cnb.cool/tmacychen/chenlongos-aka-sim:latest"
TAG_NAME="aka-sim:dev"
CONTAINER_NAME="aka-sim-dev"
HTTP_PORT=80
FALLBACK_PORT=5000

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

check_docker() {
    if ! command -v docker &>/dev/null; then
        err "未检测到 Docker，请先安装 Docker 20.10+"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        err "Docker 守护进程未运行，请先启动 Docker"
        exit 1
    fi
}

# ---- 停止容器 ----
do_stop() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "停止容器 ${CONTAINER_NAME} ..."
        docker stop "${CONTAINER_NAME}"
        ok "容器已停止"
    else
        warn "容器 ${CONTAINER_NAME} 未在运行"
    fi
}

# ---- 删除容器 ----
do_rm() {
    do_stop 2>/dev/null || true
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        info "删除容器 ${CONTAINER_NAME} ..."
        docker rm "${CONTAINER_NAME}"
        ok "容器已删除"
    fi
}

# ---- 启动容器 ----
do_start() {
    check_docker

    # 拉取镜像
    info "拉取预构建镜像 ${IMAGE_NAME} ..."
    if docker pull "${IMAGE_NAME}"; then
        ok "镜像拉取成功"
    else
        warn "镜像拉取失败，尝试使用本地已存在的镜像..."
    fi

    docker tag "${IMAGE_NAME}" "${TAG_NAME}" 2>/dev/null || true
    ok "镜像标签: ${TAG_NAME}"

    # 清理旧容器
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "容器 ${CONTAINER_NAME} 已存在，正在停止并删除..."
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
        ok "旧容器已清理"
    fi

    # 检查端口
    check_port() {
        if ss -tlnp 2>/dev/null | grep -q ":${1} "; then
            return 1
        fi
        return 0
    }

    PORT_FLAG="-p ${HTTP_PORT}:80 -p ${FALLBACK_PORT}:5000"
    if ! check_port ${HTTP_PORT}; then
        warn "端口 ${HTTP_PORT} 已被占用，改用 8080"
        PORT_FLAG="-p 8080:80 -p ${FALLBACK_PORT}:5000"
    fi
    if ! check_port ${FALLBACK_PORT}; then
        warn "端口 ${FALLBACK_PORT} 已被占用，改用 5001"
        PORT_FLAG="${PORT_FLAG/5000:5000/5001:5000}"
    fi

    info "启动容器 ${CONTAINER_NAME} ..."
    docker run -d --name "${CONTAINER_NAME}" \
      ${PORT_FLAG} \
      -v "$(pwd)":/app \
      -e PYTHONPATH=/app/backend \
      -w /app \
      "${TAG_NAME}"

    ok "容器已启动"

    # 安装前端依赖并构建
    info "安装前端依赖并构建..."
    if docker exec "${CONTAINER_NAME}" bash -c "cd /app/frontend && npm install && npm run build"; then
        ok "前端构建成功"
    else
        warn "前端构建失败，请检查: docker logs ${CONTAINER_NAME}"
    fi

    # 等待服务就绪
    info "等待服务启动..."
    sleep 3

    if check_port ${HTTP_PORT}; then
        ACCESS_PORT=${HTTP_PORT}
    elif [[ "${PORT_FLAG}" == *"8080"* ]]; then
        ACCESS_PORT=8080
    else
        ACCESS_PORT=${FALLBACK_PORT}
    fi

    for i in $(seq 1 10); do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ACCESS_PORT}" 2>/dev/null | grep -q "200\|302\|304"; then
            ok "服务已就绪"
            break
        fi
        if [ $i -eq 10 ]; then
            warn "服务启动超时，请手动检查: docker logs ${CONTAINER_NAME}"
        fi
        sleep 2
    done

    echo ""
    echo "========================================="
    echo -e "  ${GREEN}AKA-Sim 已成功启动!${NC}"
    echo "========================================="
    echo ""
    echo -e "  ${CYAN}前端页面:${NC}  http://localhost:${ACCESS_PORT}"
    echo -e "  ${CYAN}模拟器:${NC}    http://localhost:${ACCESS_PORT}/sim"
    echo -e "  ${CYAN}3D模拟器:${NC}  http://localhost:${ACCESS_PORT}/sim3d"
    echo -e "  ${CYAN}API:${NC}       http://localhost:${ACCESS_PORT}/api/datasets"
    echo ""
    echo -e "  查看日志: ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
    echo -e "  停止服务: ${YELLOW}./setup.sh stop${NC}"
    echo -e "  重启服务: ${YELLOW}./setup.sh restart${NC}"
    echo ""
}

# ---- 重启容器 ----
do_restart() {
    check_docker
    do_stop 2>/dev/null || true
    do_start
}

# ---- 显示状态 ----
do_status() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        ok "容器 ${CONTAINER_NAME} 正在运行"
        docker port "${CONTAINER_NAME}" 2>/dev/null | while read line; do
            info "端口映射: ${line}"
        done
    else
        warn "容器 ${CONTAINER_NAME} 未在运行"
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            info "容器已停止，可通过 ./setup.sh start 启动"
        fi
    fi
}

# ---- 查看日志 ----
do_logs() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker logs -f "${CONTAINER_NAME}"
    else
        err "容器 ${CONTAINER_NAME} 未在运行"
        exit 1
    fi
}

# ---- 用法 ----
usage() {
    echo ""
    echo -e "${CYAN}用法:${NC} ./setup.sh [命令]"
    echo ""
    echo -e "  ${GREEN}start${NC}       启动服务（默认）"
    echo -e "  ${GREEN}stop${NC}        停止服务"
    echo -e "  ${GREEN}restart${NC}     重启服务"
    echo -e "  ${GREEN}status${NC}      查看服务状态"
    echo -e "  ${GREEN}logs${NC}        查看实时日志"
    echo -e "  ${GREEN}rm${NC}          停止并删除容器"
    echo ""
}

# ---- 入口 ----
case "${1:-start}" in
    start)   do_start ;;
    stop)    check_docker; do_stop ;;
    restart) do_restart ;;
    status)  check_docker; do_status ;;
    logs)    check_docker; do_logs ;;
    rm)      check_docker; do_rm ;;
    -h|--help|help) usage ;;
    *)       err "未知命令: $1"; usage; exit 1 ;;
esac
