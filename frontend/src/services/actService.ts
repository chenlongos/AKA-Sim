import * as tf from '@tensorflow/tfjs';
import { Episode } from '../types';

export const CHUNK_SIZE = 10;

/** Maximum number of frames to use for training (prevent browser OOM) */
const MAX_TRAINING_FRAMES = 300;

/** Delay helper to yield to the event loop */
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

export const actService = {
    CHUNK_SIZE,

    async prepareTrainingData(episodes: Episode[]) {
        const allItems: { image: number[][][], state: number[], chunk: number[] }[] = [];

        // O(n) single pass through all episodes
        for (const ep of episodes) {
            for (let i = 0; i < ep.length; i++) {
                const frame = ep[i];
                if (!frame.image) continue;

                const chunk: number[] = [];
                for (let k = 0; k < CHUNK_SIZE; k++) {
                    const futureIndex = Math.min(i + k, ep.length - 1);
                    const futureFrame = ep[futureIndex];
                    const action = futureFrame?.action;
                    chunk.push(typeof action?.[0] === 'number' ? action[0] : 0);
                    chunk.push(typeof action?.[1] === 'number' ? action[1] : 0);
                }

                allItems.push({
                    image: frame.image,
                    state: frame.state,
                    chunk,
                });
            }
        }

        // Yield to keep UI responsive during data prep
        await tick();

        const totalFrames = allItems.length;

        // Subsample if too many frames
        let items = allItems;
        if (totalFrames > MAX_TRAINING_FRAMES) {
            const step = totalFrames / MAX_TRAINING_FRAMES;
            items = [];
            for (let i = 0; i < MAX_TRAINING_FRAMES; i++) {
                items.push(allItems[Math.floor(i * step)]);
            }
        }

        const imageInputs = items.map(it => it.image);
        const stateInputs = items.map(it => it.state);
        const labelChunks = items.map(it => it.chunk);

        return { imageInputs, stateInputs, labelChunks, totalFrames, usedFrames: items.length };
    },

    createModel() {
        // Simple dense-only model - much faster than conv2d in browser,
        // avoids WebGL shader compilation freeze
        const imageInput = tf.input({ shape: [64 * 64 * 3] }); // flattened image
        const stateInput = tf.input({ shape: [14] });

        const i1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(imageInput) as tf.SymbolicTensor;
        const s1 = tf.layers.dense({ units: 32, activation: 'relu' }).apply(stateInput) as tf.SymbolicTensor;

        const concatenated = tf.layers.concatenate().apply([i1, s1]) as tf.SymbolicTensor;

        const d1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(concatenated) as tf.SymbolicTensor;
        const d2 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(d1) as tf.SymbolicTensor;
        const output = tf.layers.dense({ units: CHUNK_SIZE * 2, activation: 'linear' }).apply(d2) as tf.SymbolicTensor;

        const model = tf.model({ inputs: [imageInput, stateInput], outputs: output });
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        return model;
    },

    async trainModel(
        model: tf.LayersModel,
        data: { imageInputs: number[][][][], stateInputs: number[][], labelChunks: number[][], totalFrames?: number, usedFrames?: number },
        onEpochEnd: (epoch: number, logs: tf.Logs | undefined) => void,
        onLog: (msg: string) => void
    ) {
        // Save current backend and switch to CPU to avoid WebGL shader compilation freeze
        const prevBackend = tf.getBackend();
        onLog(`Current TF.js backend: ${prevBackend}, switching to 'cpu' for training...`);
        await tf.setBackend('cpu');
        await tf.ready();
        onLog('CPU backend ready.');

        await tick(); // yield before heavy computation

        // Flatten images for the dense-only model input: [N, 64, 64, 3] -> [N, 64*64*3]
        const flatImages: number[][] = data.imageInputs.map(img => {
            const flat: number[] = [];
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    flat.push(img[y][x][0], img[y][x][1], img[y][x][2]);
                }
            }
            return flat;
        });

        onLog(`Flattened ${flatImages.length} images to ${flatImages[0].length}-dim vectors.`);
        await tick();

        let xsImage: tf.Tensor2D | null = null;
        let xsState: tf.Tensor2D | null = null;
        let ys: tf.Tensor2D | null = null;

        try {
            onLog('Creating tensors...');
            xsImage = tf.tensor2d(flatImages);
            xsState = tf.tensor2d(data.stateInputs);
            ys = tf.tensor2d(data.labelChunks);

            await tick();
            onLog('Tensors created. Starting model.fit()...');

            const totalEpochs = 20;
            const batchSize = 16;

            await model.fit([xsImage, xsState], ys, {
                epochs: totalEpochs,
                batchSize,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        onEpochEnd(epoch, logs);
                        // Yield after each epoch to keep UI responsive
                        return new Promise(resolve => setTimeout(resolve, 0));
                    },
                    onBatchEnd: (batch) => {
                        // Yield every batch to prevent UI freeze on CPU backend
                        if (batch % 5 === 0) {
                            return new Promise(resolve => setTimeout(resolve, 0));
                        }
                    }
                }
            });

            onLog('Training complete!');
        } finally {
            if (xsImage) xsImage.dispose();
            if (xsState) xsState.dispose();
            if (ys) ys.dispose();

            // Restore previous backend (WebGL) for inference
            try {
                await tf.setBackend(prevBackend);
                await tf.ready();
                onLog(`Restored backend to: ${tf.getBackend()}`);
            } catch {
                onLog('Could not restore WebGL backend, staying on CPU.');
            }
        }
    },

    predict(model: tf.LayersModel, image: number[][][], state: number[]) {
        return tf.tidy(() => {
            // Flatten image to match model input shape [64*64*3]
            const flatImage: number[] = [];
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    flatImage.push(image[y][x][0], image[y][x][1], image[y][x][2]);
                }
            }

            const imageTensor = tf.tensor2d([flatImage]);
            const stateTensor = tf.tensor2d([state]);
            const prediction = model.predict([imageTensor, stateTensor]) as tf.Tensor;
            return prediction.dataSync();
        });
    }
};
