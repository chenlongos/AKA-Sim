使用 HTTP 长轮询而非原生 WebSocket 存在以下问题：

## 主要缺点

### 1. **性能开销大**
```
HTTP 长轮询:
  每次请求/响应都需完整的 HTTP 头 (约 1KB)
  20FPS 控制频率 = 每秒 20 次请求/响应对
  → 每秒网络流量 ~40KB，带宽浪费

WebSocket:
  握手后仅需 2-14 字节帧头
  → 同频率下流量 < 1KB，节省 95%+ 带宽
```

### 2. **延迟更高**
```
轮询延迟 = RTT + 服务端排队时间
  轮询间隔: 50ms (20FPS)
  + 网络往返: 10-50ms
  → 总延迟 60-100ms

WebSocket: 持久连接，消息即时推送
  → 延迟 ~10-20ms
```

### 3. **连接不稳定**
- 长轮询连接超时需要重建（浏览器/代理超时通常 30-120s）
- 网络波动时频繁断线重连，用户体验差

### 4. **服务器压力大**
```
长轮询: 每个客户端占用 1 个 HTTP 线程/连接池槽位
  高并发时易耗尽连接池

WebSocket: 建立后占用极低资源
  适合多客户端场景
```

---

## 当前项目中的影响

| 场景 | 长轮询 | WebSocket |
|------|--------|-----------|
| **20FPS 控制指令** | ✅ 可用，但延迟高、带宽浪费 | ✅ 理想选择 |
| **推理 5Hz 轮询** | 已走 REST，不受影响 | — |
| **训练状态 1Hz 轮询** | 已走 REST，不受影响 | — |

> 当前配置的 **唯一合理理由** 是兼容性：某些公司防火墙或代理可能屏蔽原生 WebSocket，长轮询能保证功能可用。

---

## 改进建议

### 1. 优先使用 WebSocket，降级长轮询

```typescript
// /workspace/frontend/src/api/socket.ts
export const socket = io({
    path: "/socket.io",
    transports: ["websocket", "polling"],  // 优先 WebSocket，失败时自动降级
    upgrade: true,                         // 允许升级
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 500,
    timeout: 20000
});
```

### 2. 添加传输检测提示

```typescript
socket.on("connect", () => {
    console.log(`Socket connected via ${socket.io.engine.transport.name}`);
    if (socket.io.engine.transport.name === "polling") {
        console.warn("⚠️ Using long polling instead of WebSocket (higher latency)");
    }
});
```

---

## 总结

| 评估维度 | 当前配置 | 推荐 |
|---------|---------|------|
| 性能 | ⚠️ 中等 (高延迟、高带宽) | ✅ 优秀 |
| 稳定性 | ⚠️ 中等 (频繁重连) | ✅ 优秀 |
| 兼容性 | ✅ 极好 | ✅ 良好 (降级机制) |
| 适用场景 | 低频控制、受限网络 | 高频实时控制 |

**当前项目问题不大**：20FPS 控制频率不算极端，但改用 `["websocket", "polling"]` 优先 WebSocket 能明显提升响应速度，同时保留降级保底。