import {io} from "socket.io-client";

// 连接到后端服务器
export const socket = io("http://localhost:8000", {
    path: "/socket.io",
    transports: ["polling", "websocket"],
    upgrade: true,
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 500,
    timeout: 20000
});

export const sendAction = (action: string) => {
    socket.emit('action', action);
}

export const resetCar = () => {
    socket.emit('reset_car_state');
}

export const getCarState = () => {
    socket.emit('get_car_state');
}

export const actInfer = (payload: Record<string, unknown>) => {
    socket.emit('act_infer', payload);
}

export const saveDataset = async (payload: Record<string, unknown>) => {
    const res = await fetch(`http://localhost:8000/api/dataset`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    })
    return res.json()
}
