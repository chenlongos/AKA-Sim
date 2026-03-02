import {io} from "socket.io-client";

// 连接到后端服务器（使用相对路径，通过 Vite 代理）
export const socket = io("", {
    path: "/socket.io",
    transports: ["polling", "websocket"],
    upgrade: true,
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 500,
    timeout: 20000
});

// 发送当前按下的按键列表
export const sendActions = (actions: string[]) => {
    socket.emit('action', actions);
}

export const resetCar = () => {
    socket.emit('reset_car_state');
}

export const getCarState = () => {
    socket.emit('get_car_state');
}

// 发送图像数据用于训练数据采集
export const sendImageData = (imageData: string, actions: string[]) => {
    socket.emit('collect_data', {
        image: imageData,
        actions: actions
    });
}

// 开始/停止数据采集
export const setDataCollection = (enabled: boolean) => {
    socket.emit('set_collection', enabled);
}

// 训练相关
export const startTraining = (params: {
    data_dir?: string;
    output_dir?: string;
    epochs?: number;
    batch_size?: number;
    lr?: number;
}) => {
    return fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    }).then(res => res.json());
};

export const getTrainingStatus = () => {
    return fetch('/api/train/status').then(res => res.json());
};

export const stopTraining = () => {
    return fetch('/api/train/stop', { method: 'POST' }).then(res => res.json());
};

// 监听训练进度
export const onTrainingProgress = (callback: (data: {
    is_running: boolean;
    epoch: number;
    total_epochs: number;
    loss: number;
    progress: number;
}) => void) => {
    socket.on('training_progress', callback);
    return () => socket.off('training_progress');
};