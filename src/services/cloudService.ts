import { CloudModel, CloudDataset, CloudTrainingStatus } from '../types';

const baseUrl = "";

export const cloudService = {
    async fetchModels(): Promise<CloudModel[]> {
        try {
            const res = await fetch(baseUrl + '/api/models', { mode: 'cors' });
            const data = await res.json();
            return data.models || [];
        } catch (e) {
            console.error('Failed to fetch models', e);
            return [];
        }
    },

    async fetchDatasets(): Promise<CloudDataset[]> {
        try {
            const res = await fetch(baseUrl + '/api/datasets', { mode: 'cors' });
            const data = await res.json();
            return data.datasets || [];
        } catch (e) {
            console.error('Failed to fetch datasets', e);
            return [];
        }
    },

    async saveDataset(dataset: any): Promise<boolean> {
        try {
            const res = await fetch(baseUrl + '/api/dataset', {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataset)
            });
            const data = await res.json();
            return data.status === 'success';
        } catch (e) {
            console.error('Failed to upload dataset', e);
            return false;
        }
    },

    async startTraining(datasetPath: string): Promise<boolean> {
        try {
            const res = await fetch(baseUrl + '/api/train/start', {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_path: datasetPath
                })
            });
            const data = await res.json();
            return res.ok;
        } catch (e) {
            console.error('Failed to start cloud training', e);
            return false;
        }
    },

    async getTrainingStatus(): Promise<CloudTrainingStatus | null> {
        try {
            const res = await fetch(baseUrl + '/api/train/status', { mode: 'cors' });
            const data = await res.json();
            return data;
        } catch (e) {
            return null;
        }
    },

    async startInference(modelId: string): Promise<boolean> {
        try {
            const res = await fetch(baseUrl + '/api/infer/start', {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            });
            const data = await res.json();
            return res.ok;
        } catch (e) {
            console.error('Failed to start cloud inference', e);
            return false;
        }
    },

    async stopInference(): Promise<boolean> {
        try {
            await fetch(baseUrl + '/api/infer/stop', { method: 'POST', mode: 'cors' });
            return true;
        } catch (e) {
            return false;
        }
    },

    async runInferenceStep(state: number[], envState: number[]): Promise<any> {
        try {
            const res = await fetch(baseUrl + '/api/infer/step', {
                method: 'POST',
                mode: 'cors',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state: state,
                    env_state: envState
                })
            });
            const data = await res.json();
            if (data.action) {
                return data.action;
            }
            return null;
        } catch (e) {
            console.error('Cloud inference step failed', e);
            return null;
        }
    }
};
