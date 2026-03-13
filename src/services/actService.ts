import * as tf from '@tensorflow/tfjs';
import { Episode } from '../types';

export const CHUNK_SIZE = 10;

export const actService = {
    CHUNK_SIZE,
    prepareTrainingData(episodes: Episode[]) {
        const imageInputs: number[][][][] = [];
        const stateInputs: number[][] = [];
        const labelChunks: number[][] = [];
        
        episodes.forEach(episode => {
            for (let i = 0; i < episode.length; i++) {
                const frame = episode[i];
                if (!frame.image) continue;

                const chunk: number[] = [];
                for (let k = 0; k < CHUNK_SIZE; k++) {
                    const futureIndex = Math.min(i + k, episode.length - 1);
                    const futureFrame = episode[futureIndex];
                    chunk.push(futureFrame.action[0]);
                    chunk.push(futureFrame.action[1]);
                }

                imageInputs.push(frame.image);
                stateInputs.push(frame.state);
                labelChunks.push(chunk);
            }
        });

        return { imageInputs, stateInputs, labelChunks };
    },

    createModel() {
        const imageInput = tf.input({shape: [64, 64, 3]});
        const stateInput = tf.input({shape: [14]});

        const h1 = tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}).apply(imageInput) as tf.SymbolicTensor;
        const h2 = tf.layers.maxPooling2d({poolSize: 2}).apply(h1) as tf.SymbolicTensor;
        const h3 = tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu'}).apply(h2) as tf.SymbolicTensor;
        const h4 = tf.layers.maxPooling2d({poolSize: 2}).apply(h3) as tf.SymbolicTensor;
        const h5 = tf.layers.flatten().apply(h4) as tf.SymbolicTensor;
        
        const s1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(stateInput) as tf.SymbolicTensor;
        
        const concatenated = tf.layers.concatenate().apply([h5, s1]) as tf.SymbolicTensor;
        
        const d1 = tf.layers.dense({units: 128, activation: 'relu'}).apply(concatenated) as tf.SymbolicTensor;
        const output = tf.layers.dense({units: CHUNK_SIZE * 2, activation: 'linear'}).apply(d1) as tf.SymbolicTensor;
        
        const model = tf.model({inputs: [imageInput, stateInput], outputs: output});
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        
        return model;
    },

    async trainModel(
        model: tf.LayersModel, 
        data: { imageInputs: number[][][][], stateInputs: number[][], labelChunks: number[][] },
        onEpochEnd: (epoch: number, logs: tf.Logs | undefined) => void
    ) {
        const xsImage = tf.tensor4d(data.imageInputs);
        const xsState = tf.tensor2d(data.stateInputs);
        const ys = tf.tensor2d(data.labelChunks);

        await model.fit([xsImage, xsState], ys, {
            epochs: 50,
            batchSize: 32,
            shuffle: true,
            callbacks: {
                onEpochEnd
            }
        });

        xsImage.dispose();
        xsState.dispose();
        ys.dispose();
    },

    predict(model: tf.LayersModel, image: number[][][], state: number[]) {
        return tf.tidy(() => {
            const imageTensor = tf.tensor4d([image]);
            const stateTensor = tf.tensor2d([state]);
            const prediction = model.predict([imageTensor, stateTensor]) as tf.Tensor;
            return prediction.dataSync();
        });
    }
};
