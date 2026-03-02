"""
ACT 模型 PyTorch 版本测试
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_act import (
    ACTModel,
    ACTLoss,
    ACTTrainer,
    create_act_model,
)
from ACTDataset import ACTDataset
from configuration_act import ACTConfig


def test_act_config():
    """测试配置类"""
    print("测试 ACT 配置...")
    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
    )
    assert config.state_dim == 7
    assert config.action_dim == 7
    assert config.action_chunk_size == 16
    print("配置测试通过!")


def test_act_model_forward():
    """测试 ACT 模型前向传播"""
    print("\n测试 ACT 模型前向传播...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    model = ACTModel(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    images = torch.randn(batch_size, 1, 3, 224, 224)
    state = torch.randn(batch_size, 7)
    action_target = torch.randn(batch_size, 16, 7)

    output = model(images, state, action_target)

    assert output["action"].shape == (batch_size, 16, 7), f"Expected shape {(batch_size, 16, 7)}, got {output['action'].shape}"
    print(f"前向传播测试通过! 输出形状: {output['action'].shape}")


def test_act_model_get_action():
    """测试获取动作"""
    print("\n测试获取动作...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
    )

    model = ACTModel(config)
    model.eval()

    images = torch.randn(1, 1, 3, 224, 224)
    state = torch.randn(1, 7)

    with torch.no_grad():
        action = model.get_action(images, state)

    assert action.shape == (1, 16, 7), f"Expected shape {(1, 16, 7)}, got {action.shape}"
    print(f"获取动作测试通过! 动作形状: {action.shape}")


def test_act_loss():
    """测试损失函数"""
    print("\n测试损失函数...")

    loss_fn = ACTLoss(action_chunk_size=16)

    pred_action = torch.randn(4, 16, 7)
    target_action = torch.randn(4, 16, 7)

    loss_dict = loss_fn(pred_action, target_action)

    assert "loss" in loss_dict
    assert "first_step_loss" in loss_dict
    assert "last_step_loss" in loss_dict
    print(f"损失测试通过! 损失值: {loss_dict['loss'].item():.4f}")


def test_act_dataset():
    """测试数据集"""
    print("\n测试数据集...")

    # 创建模拟数据 - 使用与 ACTDataset 兼容的格式
    # action 应该是 [total_timesteps, action_dim]，每个时间步一个动作
    num_samples = 100
    action_chunk_size = 16
    data = {
        "observation.image": torch.randn(num_samples, 1, 3, 224, 224),
        "observation.state": torch.randn(num_samples, 7),
        "action": torch.randn(num_samples, 7),  # [total_timesteps, action_dim]
    }

    dataset = ACTDataset(data, action_chunk_size=16)

    assert len(dataset) == num_samples - 16 + 1, f"Expected length {num_samples - 16 + 1}, got {len(dataset)}"

    sample = dataset[0]
    assert "observation" in sample
    assert "action" in sample
    assert sample["action"].shape == (16, 7)

    print(f"数据集测试通过! 数据集大小: {len(dataset)}")


def test_act_trainer():
    """测试训练器"""
    print("\n测试训练器...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    model = ACTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = ACTTrainer(model, optimizer)

    # 创建模拟批数据
    batch = {
        "observation": {
            "image": torch.randn(4, 1, 3, 224, 224),
            "state": torch.randn(4, 7),
        },
        "action": torch.randn(4, 16, 7),
    }

    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    print(f"训练器测试通过! 损失: {metrics['loss']:.4f}")


def test_create_act_model():
    """测试便捷创建函数"""
    print("\n测试便捷创建函数...")

    model = create_act_model(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
    )

    assert isinstance(model, ACTModel)
    print(f"便捷创建测试通过!")


def test_training_loop():
    """测试完整训练循环"""
    print("\n测试完整训练循环...")

    config = ACTConfig(
        state_dim=7,
        action_dim=7,
        action_chunk_size=16,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )

    model = ACTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = ACTTrainer(model, optimizer)

    # 创建模拟数据 - action 应该是 [total_timesteps, action_dim]
    num_samples = 50
    data = {
        "observation.image": torch.randn(num_samples, 1, 3, 224, 224),
        "observation.state": torch.randn(num_samples, 7),
        "action": torch.randn(num_samples, 7),  # [total_timesteps, action_dim]
    }

    dataset = ACTDataset(data, action_chunk_size=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # 训练几个 epoch
    for epoch in range(3):
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["loss"]
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1}: 损失 = {avg_loss:.4f}")

    print("完整训练循环测试通过!")


def main():
    """运行所有测试"""
    print("=" * 50)
    print("开始测试 ACT PyTorch 实现")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    test_act_config()
    test_act_model_forward()
    test_act_model_get_action()
    test_act_loss()
    test_act_dataset()
    test_act_trainer()
    test_create_act_model()
    test_training_loop()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
