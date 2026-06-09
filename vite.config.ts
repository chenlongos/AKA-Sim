import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    base: './',
    plugins: [react(), tailwindcss(), viteSingleFile()],
    build: {
      cssCodeSplit: false,       // 不拆分 CSS
      assetsInlineLimit: 1024 * 1024 * 10, // 强制内联所有资源（10MB）
      // 👇 删掉了冲突的 manualChunks
      rollupOptions: {
        output: {
          // 这里删除 manualChunks 配置
        }
      },
    },
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      proxy: {
        "/api": {
          target: "http://127.0.0.1:5000",
          changeOrigin: true,
        },
        "/socket.io": {
          target: "http://127.0.0.1:5000",
          ws: false,
          changeOrigin: true,
        },
      },
    },
  };
});