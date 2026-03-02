import {useCallback, useEffect, useRef, useState} from "react"
import {socket, sendActions, resetCar, getCarState, sendImageData, setDataCollection, startTraining, getTrainingStatus, stopTraining} from "../api/socket";

const MAP_W = 800;
const MAP_H = 600;

const FPS = 30
const SEND_INTERVAL = 50 // 发送控制指令间隔(ms)
const frameInterval = 1000 / FPS

// 小车状态类型
interface CarState {
    x: number;
    y: number;
    angle: number;
    speed: number;
    maxSpeed: number;
    acceleration: number;
    friction: number;
    rotationSpeed: number;
}

const SimPage = () => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const fpvRef = useRef<HTMLCanvasElement | null>(null);
    const keys = useRef<Record<string, boolean>>({})
    const [carState, setCarState] = useState<CarState>({
        x: 400,
        y: 300,
        angle: -Math.PI / 2,
        speed: 0,
        maxSpeed: 5,
        acceleration: 0.2,
        friction: 0.95,
        rotationSpeed: 0.05
    })
    const [isCollecting, setIsCollecting] = useState(false)
    const [collectedCount, setCollectedCount] = useState(0)
    const [isTraining, setIsTraining] = useState(false)
    const [trainingProgress, setTrainingProgress] = useState({ epoch: 0, total_epochs: 50, loss: 0, progress: 0 })

    // 监听后端车辆状态更新
    useEffect(() => {
        // 连接 Socket.IO
        socket.connect()

        // 监听连接
        socket.on("connected", (data) => {
            console.log("Connected:", data)
            // 连接后获取初始状态
            getCarState()
        })

        // 监听车辆状态更新
        socket.on("car_state_update", (state: CarState) => {
            setCarState(state)
        })

        // 监听采集计数更新
        socket.on("collection_count", (data: { count: number; exported?: boolean; output_path?: string; error?: string }) => {
            setCollectedCount(data.count)
            if (data.exported) {
                alert(`数据已导出到: ${data.output_path}`)
            } else if (data.error) {
                alert(`导出失败: ${data.error}`)
            }
        })

        // 监听训练进度
        socket.on("training_progress", (data: { is_running: boolean; epoch: number; total_epochs: number; loss: number; progress: number }) => {
            setIsTraining(data.is_running)
            setTrainingProgress({
                epoch: data.epoch,
                total_epochs: data.total_epochs,
                loss: data.loss,
                progress: data.progress
            })
        })

        // 获取初始训练状态
        getTrainingStatus().then(data => {
            setIsTraining(data.is_running)
            setTrainingProgress({
                epoch: data.epoch,
                total_epochs: data.total_epochs,
                loss: data.loss,
                progress: data.progress
            })
        })

        return () => {
            socket.off("connected")
            socket.off("car_state_update")
            socket.off("collection_count")
            socket.off("training_progress")
            socket.disconnect()
        }
    }, [])

    const sendCommand = (cmd: string) => {
        // 发送动作到后端
        sendActions([cmd])
    }

    const toggleCollection = () => {
        const newState = !isCollecting
        setIsCollecting(newState)
        setDataCollection(newState)
    }

    const handleStartTraining = async () => {
        try {
            const result = await startTraining({
                data_dir: 'dataset',
                output_dir: 'checkpoints',
                epochs: 50,
                batch_size: 8,
                lr: 1e-4,
            })
            if (!result.success) {
                alert(result.message)
            }
        } catch (e) {
            alert('启动训练失败')
        }
    }

    const handleStopTraining = async () => {
        try {
            await stopTraining()
        } catch (e) {
            alert('停止训练失败')
        }
    }

    const drawGrid = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
        ctx.strokeStyle = '#e0e0e0'
        ctx.lineWidth = 1
        const gridSize = 50

        ctx.beginPath()
        for (let x = 0; x <= w; x += gridSize) {
            ctx.moveTo(x, 0)
            ctx.lineTo(x, h)
        }
        for (let y = 0; y <= h; y += gridSize) {
            ctx.moveTo(0, y)
            ctx.lineTo(w, y)
        }
        ctx.stroke()
    }, [])

    const drawCarBody = useCallback((ctx: CanvasRenderingContext2D, x: number, y: number, angle: number) => {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.fillStyle = 'blue';
        ctx.fillRect(-20, -10, 40, 20);
        ctx.fillStyle = 'yellow';
        ctx.beginPath();
        ctx.arc(15, -6, 3, 0, Math.PI * 2);
        ctx.arc(15, 6, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#2c3e50'
        ctx.fillRect(5, -8, 10, 16)
        ctx.restore();
    }, [])

    const drawTopDown = useCallback((ctx: CanvasRenderingContext2D) => {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        drawGrid(ctx, ctx.canvas.width, ctx.canvas.height)

        ctx.save()

        // 使用后端的车辆状态
        drawCarBody(ctx, carState.x, carState.y, carState.angle)

        ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        ctx.beginPath();
        ctx.moveTo(carState.x, carState.y);
        ctx.lineTo(carState.x + Math.cos(carState.angle - Math.PI / 6) * 100, carState.y + Math.sin(carState.angle - Math.PI / 6) * 100);
        ctx.moveTo(carState.x, carState.y);
        ctx.lineTo(carState.x + Math.cos(carState.angle + Math.PI / 6) * 100, carState.y + Math.sin(carState.angle + Math.PI / 6) * 100);
        ctx.stroke();

        ctx.restore()
    }, [carState, drawCarBody, drawGrid])


    const getRaySegmentIntersection = (rx: number, ry: number, rdx: number, rdy: number, wall: {
        x1: number,
        y1: number,
        x2: number,
        y2: number
    }) => {
        const {x1, y1, x2, y2} = wall;
        const v1x = x1 - rx;
        const v1y = y1 - ry;
        const v2x = x2 - x1;
        const v2y = y2 - y1;
        const v3x = -rdx; // 射线方向反转
        const v3y = -rdy;

        const cross = v2x * v3y - v2y * v3x;
        if (Math.abs(cross) < 0.0001) return null; // 平行

        const t1 = (v2x * v1y - v2y * v1x) / cross; // 射线距离
        const t2 = (v3x * v1y - v3y * v1x) / cross; // 线段比例 (0~1)

        // t1 > 0 代表射线前方，t2 在 0~1 代表交点在线段上
        if (t1 > 0 && t2 >= 0 && t2 <= 1) {
            return t1;
        }
        return null;
    };

    const castRay = (sx: number, sy: number, angle: number) => {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        let minDist = Infinity;
        let hitColor = null;

        // 将所有障碍物转换为线段进行检测
        const boundaries = [
            {x1: 0, y1: 0, x2: MAP_W, y2: 0, color: '#333'}, // 上墙
            {x1: MAP_W, y1: 0, x2: MAP_W, y2: MAP_H, color: '#333'}, // 右墙
            {x1: MAP_W, y1: MAP_H, x2: 0, y2: MAP_H, color: '#333'}, // 下墙
            {x1: 0, y1: MAP_H, x2: 0, y2: 0, color: '#333'}  // 左墙
        ];

        // 检测射线与每一条线段的交点
        boundaries.forEach(wall => {
            const dist = getRaySegmentIntersection(sx, sy, cos, sin, wall);
            if (dist !== null && dist < minDist) {
                minDist = dist;
                hitColor = wall.color;
            }
        });

        return minDist === Infinity ? null : {distance: minDist, color: hitColor};
    };

    const drawFirstPerson = useCallback((ctx: CanvasRenderingContext2D) => {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const {x, y, angle} = carState;

        // 天空和地面
        ctx.fillStyle = '#87CEEB'; // 天空蓝
        ctx.fillRect(0, 0, w, h / 2);
        ctx.fillStyle = '#7f8c8d'; // 地面灰
        ctx.fillRect(0, h / 2, w, h / 2);

        // 参数
        const fov = Math.PI / 3; // 60度视野
        const rayCount = w / 4;  // 射线数量 (为了性能，每4个像素投射一条，然后画宽一点)
        const rayWidth = w / rayCount;

        // 遍历每一条射线
        for (let i = 0; i < rayCount; i++) {
            // 当前射线角度 = 车角度 - 半个FOV + 增量
            const rayAngle = (angle + Math.PI - fov / 2) + (i / rayCount) * fov;

            // 计算这一条射线碰到了什么，以及距离是多少
            const hit = castRay(x, y, rayAngle);

            if (hit) {
                const correctedDist = hit.distance * Math.cos(rayAngle - angle);

                const wallHeight = (h * 40) / correctedDist;

                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                ctx.fillStyle = hit.color;
                ctx.globalAlpha = Math.max(0.3, 1 - correctedDist / 600);
                ctx.fillRect(i * rayWidth, (h - wallHeight) / 2, rayWidth + 1, wallHeight);
                ctx.globalAlpha = 1.0;
            }
        }
    }, [carState, castRay])


    useEffect(() => {
        const canvas = canvasRef.current
        const fpv = fpvRef.current
        if (canvas == null || fpv == null) return
        const ctxTop = canvas.getContext('2d')
        const ctxFpv = fpv.getContext('2d')

        if (ctxTop == null || ctxFpv == null) return

        ctxFpv.imageSmoothingEnabled = false;

        let animationFrameId: number

        const handleKeyDown = (e: KeyboardEvent) => {
            const active = document.activeElement as HTMLElement | null
            if (active) {
                const tag = active.tagName
                if (active.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
                    return
                }
            }
            if (e.code.startsWith("Arrow")) {
                e.preventDefault()
            }
            keys.current[e.code] = true
        }
        const handleKeyUp = (e: KeyboardEvent) => {
            const active = document.activeElement as HTMLElement | null
            if (active) {
                const tag = active.tagName
                if (active.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
                    return
                }
            }
            if (e.code.startsWith("Arrow")) {
                e.preventDefault()
            }
            keys.current[e.code] = false
        }

        // 获取当前按下的动作列表
        const getCurrentActions = (): string[] => {
            const keyMap: Record<string, string> = {
                'ArrowUp': 'forward',
                'KeyW': 'forward',
                'ArrowDown': 'backward',
                'KeyS': 'backward',
                'ArrowLeft': 'left',
                'KeyA': 'left',
                'ArrowRight': 'right',
                'KeyD': 'right',
            }

            const actions: string[] = []
            for (const [code, action] of Object.entries(keyMap)) {
                if (keys.current[code]) {
                    actions.push(action)
                }
            }
            return actions
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)

        let lastTime = 0;
        let lastSendTime = 0;
        let lastCollectTime = 0;
        const COLLECT_INTERVAL = 100 // 采集间隔(ms)，10fps

        const renderLoop = (currentTime: number) => {
            animationFrameId = window.requestAnimationFrame(renderLoop)

            const delta = currentTime - lastTime

            if (delta < frameInterval) return

            lastTime = currentTime - (delta % frameInterval)

            // 控制发送频率
            if (currentTime - lastSendTime >= SEND_INTERVAL) {
                const actions = getCurrentActions()
                sendActions(actions)
                lastSendTime = currentTime
            }

            // 如果正在采集，捕获并发送图像数据
            if (isCollecting && currentTime - lastCollectTime >= COLLECT_INTERVAL) {
                const actions = getCurrentActions()
                // 从第一人称Canvas获取图像数据
                const imageData = fpv.toDataURL('image/jpeg', 0.8)
                sendImageData(imageData, actions)
                lastCollectTime = currentTime
            }

            // 渲染
            drawTopDown(ctxTop)
            drawFirstPerson(ctxFpv)
        }

        animationFrameId = window.requestAnimationFrame(renderLoop)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)

            window.cancelAnimationFrame(animationFrameId)
        }
    }, [drawFirstPerson, drawTopDown, isCollecting])

    return (
        <div className="flex flex-col gap-3 p-4 h-screen overflow-hidden">
            <h1 className="text-center font-bold">AKA-Sim 模拟器</h1>
            <div className="flex gap-5 flex-1 items-stretch">
                <div className="w-64 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-2">
                        <div className="font-semibold">训练控制</div>
                        <div className="text-xs text-gray-600">
                            已采集样本: {collectedCount}
                        </div>
                        {!isTraining ? (
                            <button
                                onClick={handleStartTraining}
                                disabled={collectedCount === 0}
                                className={`px-3 py-1 rounded ${collectedCount > 0 ? 'bg-purple-500 text-white hover:bg-purple-600' : 'bg-gray-300 text-gray-500'}`}
                            >
                                开始训练
                            </button>
                        ) : (
                            <button
                                onClick={handleStopTraining}
                                className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                            >
                                停止训练
                            </button>
                        )}
                        {isTraining && (
                            <div className="mt-2">
                                <div className="text-xs text-gray-600">
                                    Epoch: {trainingProgress.epoch}/{trainingProgress.total_epochs}
                                </div>
                                <div className="text-xs text-gray-600">
                                    Loss: {trainingProgress.loss.toFixed(6)}
                                </div>
                                <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-purple-500 h-2 rounded-full transition-all"
                                        style={{ width: `${trainingProgress.progress * 100}%` }}
                                    />
                                </div>
                                <div className="text-xs text-gray-600 text-center mt-1">
                                    {Math.round(trainingProgress.progress * 100)}%
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                <div className="flex-1 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-3 flex flex-col gap-3 h-full">
                        <div className="font-semibold">俯视地图</div>
                        <div className="relative flex justify-center">
                            <canvas
                                ref={canvasRef}
                                width={800}
                                height={600}
                                className="bg-white block rounded"
                            />
                            <div className="absolute top-2 left-2 bg-white/85 p-1.5 rounded text-xs">
                                使用 WASD 或 方向键 移动<br/>
                                实时同步后端状态
                            </div>
                        </div>
                        <div className="flex gap-2.5 flex-wrap justify-center items-center">
                            <button onClick={() => sendCommand('forward')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 前进</button>
                            <button onClick={() => sendCommand('left')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 左转</button>
                            <button onClick={() => sendCommand('right')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 右转</button>
                            <button onClick={() => sendCommand('backward')} className="px-3 py-1 bg-blue-500 text-black rounded hover:bg-blue-600">指令: 后退</button>
                            <button onClick={() => resetCar()} className="px-3 py-1 bg-green-500 text-black rounded hover:bg-green-600">复位</button>
                            <button
                                onClick={toggleCollection}
                                className={`px-3 py-1 rounded ${isCollecting ? 'bg-red-500 text-white hover:bg-red-600' : 'bg-yellow-500 text-black hover:bg-yellow-600'}`}
                            >
                                {isCollecting ? '停止采集' : '开始采集'}
                            </button>
                            <span className="text-xs text-gray-600 ml-2">已采集: {collectedCount}</span>
                        </div>
                    </div>
                </div>
                <div className="w-90 flex flex-col h-full">
                    <div className="border-2 border-gray-800 rounded-lg bg-gray-100 p-2.5 flex flex-col gap-2 h-full text-gray-800 overflow-y-auto min-h-0">
                        <div className="font-semibold">车载摄像头</div>
                        <canvas ref={fpvRef} width={320} height={240}
                                className="bg-black border-2 border-gray-800 rounded self-center"/>
                        <div className="text-xs text-gray-600">
                            说明：右侧画面是根据左侧地图实时计算生成的伪3D视角。<br/>
                            状态来源：后端实时同步
                        </div>
                        <div className="text-xs mt-2">
                            <div className="font-semibold">当前状态:</div>
                            <div>X: {carState.x.toFixed(1)}</div>
                            <div>Y: {carState.y.toFixed(1)}</div>
                            <div>角度: {(carState.angle * 180 / Math.PI).toFixed(1)}°</div>
                            <div>速度: {carState.speed.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SimPage;
