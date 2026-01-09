"""
TerraTNT: 目标驱动的地面目标轨迹预测模型
包含CNN环境编码器、LSTM历史编码器、目标分类器和LSTM解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNEnvironmentEncoder(nn.Module):
    """CNN环境编码器 - 使用ResNet-18提取环境特征"""
    
    def __init__(self, input_channels: int = 18, feature_dim: int = 256):
        """
        Args:
            input_channels: 输入通道数 (18通道环境地图)
            feature_dim: 输出特征维度
        """
        super().__init__()
        
        # 使用预训练的ResNet-18作为backbone
        resnet = models.resnet18(pretrained=False)
        
        # 修改第一层以接受18通道输入
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 投影到特征维度
        self.fc = nn.Linear(512, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 18, 128, 128) 环境地图
            
        Returns:
            (batch, feature_dim) 环境特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class LSTMHistoryEncoder(nn.Module):
    """LSTM历史轨迹编码器"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2):
        """
        Args:
            input_dim: 输入维度 (2D坐标)
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, 2) 历史轨迹
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            (h_n, c_n): 最终隐藏状态
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class GoalClassifier(nn.Module):
    """目标分类器 - 对候选终点进行概率评分"""
    
    def __init__(self, env_feature_dim: int = 256, 
                 history_feature_dim: int = 128,
                 hidden_dim: int = 256,
                 num_goals: int = 100):
        """
        Args:
            env_feature_dim: 环境特征维度
            history_feature_dim: 历史特征维度
            hidden_dim: 隐藏层维度
            num_goals: 候选目标数量
        """
        super().__init__()
        
        self.num_goals = num_goals
        
        # 融合环境和历史特征
        self.fusion = nn.Sequential(
            nn.Linear(env_feature_dim + history_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 目标评分网络
        self.goal_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for goal coordinates
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, env_features: torch.Tensor, 
                history_features: torch.Tensor,
                candidate_goals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_features: (batch, env_feature_dim) 环境特征
            history_features: (batch, history_feature_dim) 历史特征
            candidate_goals: (batch, num_goals, 2) 候选目标坐标
            
        Returns:
            goal_probs: (batch, num_goals) 目标概率分布
        """
        batch_size = env_features.size(0)
        
        # 融合特征
        fused = torch.cat([env_features, history_features], dim=1)
        fused = self.fusion(fused)  # (batch, hidden_dim)
        
        # 扩展到所有候选目标
        fused_expanded = fused.unsqueeze(1).expand(-1, self.num_goals, -1)  # (batch, num_goals, hidden_dim)
        
        # 拼接候选目标坐标
        goal_input = torch.cat([fused_expanded, candidate_goals], dim=2)  # (batch, num_goals, hidden_dim+2)
        
        # 评分
        scores = self.goal_scorer(goal_input).squeeze(-1)  # (batch, num_goals)
        
        # Softmax得到概率分布
        goal_probs = F.softmax(scores, dim=1)
        
        return goal_probs


class HierarchicalLSTMDecoder(nn.Module):
    """层次化LSTM解码器 - 生成未来轨迹"""
    
    def __init__(self, input_dim: int = 2,
                 env_feature_dim: int = 256,
                 history_feature_dim: int = 128,
                 goal_feature_dim: int = 2,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_length: int = 60):
        """
        Args:
            input_dim: 输入维度 (2D坐标)
            env_feature_dim: 环境特征维度
            history_feature_dim: 历史特征维度
            goal_feature_dim: 目标特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            output_length: 输出序列长度
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        # 特征融合
        total_feature_dim = env_feature_dim + history_feature_dim + goal_feature_dim
        self.feature_projection = nn.Linear(total_feature_dim, hidden_dim)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=input_dim + hidden_dim,  # 当前位置 + 上下文特征
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, env_features: torch.Tensor,
                history_features: torch.Tensor,
                goal_features: torch.Tensor,
                current_pos: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                ground_truth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            env_features: (batch, env_feature_dim)
            history_features: (batch, history_feature_dim)
            goal_features: (batch, goal_feature_dim)
            current_pos: (batch, 2) 当前位置
            teacher_forcing_ratio: 教师强制比率
            ground_truth: (batch, seq_len, 2) 真实未来轨迹 (训练时使用)
            
        Returns:
            predictions: (batch, output_length, 2) 预测的未来轨迹
        """
        batch_size = env_features.size(0)
        
        # 融合所有特征
        context = torch.cat([env_features, history_features, goal_features], dim=1)
        context = self.feature_projection(context)  # (batch, hidden_dim)
        
        # 初始化LSTM隐藏状态
        h_0 = context.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c_0 = torch.zeros_like(h_0)
        
        # 解码
        predictions = []
        input_pos = current_pos  # 起始位置
        hidden = (h_0, c_0)
        
        for t in range(self.output_length):
            # 拼接输入和上下文
            context_expanded = context.unsqueeze(1)  # (batch, 1, hidden_dim)
            lstm_input = torch.cat([input_pos.unsqueeze(1), context_expanded], dim=2)  # (batch, 1, input_dim+hidden_dim)
            
            # LSTM前向
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            # 预测下一个位置
            pred = self.output_layer(lstm_out.squeeze(1))  # (batch, 2)
            predictions.append(pred)
            
            # 教师强制
            if ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_pos = ground_truth[:, t, :]
            else:
                input_pos = pred
        
        predictions = torch.stack(predictions, dim=1)  # (batch, output_length, 2)
        
        return predictions


class TerraTNT(nn.Module):
    """TerraTNT完整模型"""
    
    def __init__(self, config: dict = None, **kwargs):
        """
        Args:
            config: 配置字典，包含以下键：
                - env_channels: 环境地图通道数 (默认18)
                - env_feature_dim: 环境特征维度 (默认256)
                - history_hidden_dim: 历史编码器隐藏维度 (默认128)
                - decoder_hidden_dim: 解码器隐藏维度 (默认256)
                - num_goals: 候选目标数量 (默认100)
                - output_length/future_len: 输出轨迹长度 (默认60)
        """
        super().__init__()
        
        # 支持两种初始化方式：config字典或直接参数
        if config is None:
            config = kwargs
        
        # 提取配置参数
        env_channels = config.get('env_channels', 18)
        env_feature_dim = config.get('env_feature_dim', 256)
        history_hidden_dim = config.get('history_hidden_dim', 128)
        decoder_hidden_dim = config.get('decoder_hidden_dim', 256)
        num_goals = config.get('num_goals', config.get('num_goals', 100))
        output_length = config.get('output_length', config.get('future_len', 60))
        
        self.num_goals = num_goals
        self.output_length = output_length
        
        # 环境编码器
        self.env_encoder = CNNEnvironmentEncoder(
            input_channels=env_channels,
            feature_dim=env_feature_dim
        )
        
        # 历史轨迹编码器
        self.history_encoder = LSTMHistoryEncoder(
            input_dim=2,
            hidden_dim=history_hidden_dim,
            num_layers=2
        )
        
        # 目标分类器
        self.goal_classifier = GoalClassifier(
            env_feature_dim=env_feature_dim,
            history_feature_dim=history_hidden_dim,
            num_goals=num_goals
        )
        
        # 轨迹解码器
        self.decoder = HierarchicalLSTMDecoder(
            input_dim=2,
            env_feature_dim=env_feature_dim,
            history_feature_dim=history_hidden_dim,
            goal_feature_dim=2,
            hidden_dim=decoder_hidden_dim,
            num_layers=2,
            output_length=output_length
        )
        
    def forward(self, env_map: torch.Tensor,
                history: torch.Tensor,
                candidate_goals: torch.Tensor,
                current_pos: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                ground_truth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            env_map: (batch, 18, 128, 128) 环境地图
            history: (batch, seq_len, 2) 历史轨迹
            candidate_goals: (batch, num_goals, 2) 候选目标
            current_pos: (batch, 2) 当前位置
            teacher_forcing_ratio: 教师强制比率
            ground_truth: (batch, output_length, 2) 真实未来轨迹
            
        Returns:
            predictions: (batch, output_length, 2) 预测轨迹
            goal_probs: (batch, num_goals) 目标概率
        """
        # 编码环境
        env_features = self.env_encoder(env_map)  # (batch, env_feature_dim)
        
        # 编码历史轨迹
        history_output, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]  # 使用最后一层的隐藏状态 (batch, history_hidden_dim)
        
        # 目标分类
        goal_probs = self.goal_classifier(env_features, history_features, candidate_goals)
        
        # 选择最可能的目标
        _, top_goal_idx = torch.max(goal_probs, dim=1)
        selected_goals = candidate_goals[torch.arange(candidate_goals.size(0)), top_goal_idx]  # (batch, 2)
        
        # 解码生成轨迹
        predictions = self.decoder(
            env_features=env_features,
            history_features=history_features,
            goal_features=selected_goals,
            current_pos=current_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            ground_truth=ground_truth
        )
        
        return predictions, goal_probs
    
    def predict(self, env_map: torch.Tensor,
                history: torch.Tensor,
                candidate_goals: torch.Tensor,
                current_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理模式
        
        Returns:
            predictions: (batch, output_length, 2) 预测轨迹
            goal_probs: (batch, num_goals) 目标概率
            selected_goals: (batch, 2) 选择的目标
        """
        self.eval()
        with torch.no_grad():
            predictions, goal_probs = self.forward(
                env_map=env_map,
                history=history,
                candidate_goals=candidate_goals,
                current_pos=current_pos,
                teacher_forcing_ratio=0.0  # 推理时不使用教师强制
            )
            
            _, top_goal_idx = torch.max(goal_probs, dim=1)
            selected_goals = candidate_goals[torch.arange(candidate_goals.size(0)), top_goal_idx]
            
        return predictions, goal_probs, selected_goals


def test_model():
    """测试模型"""
    logger.info("=" * 60)
    logger.info("测试TerraTNT模型")
    logger.info("=" * 60)
    
    # 创建模型
    model = TerraTNT(
        env_channels=18,
        env_feature_dim=256,
        history_hidden_dim=128,
        decoder_hidden_dim=256,
        num_goals=100,
        output_length=60
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n模型参数:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    env_map = torch.randn(batch_size, 18, 128, 128)
    history = torch.randn(batch_size, 10, 2)
    candidate_goals = torch.randn(batch_size, 100, 2)
    current_pos = torch.randn(batch_size, 2)
    ground_truth = torch.randn(batch_size, 60, 2)
    
    logger.info(f"\n测试前向传播:")
    logger.info(f"  输入形状:")
    logger.info(f"    env_map: {env_map.shape}")
    logger.info(f"    history: {history.shape}")
    logger.info(f"    candidate_goals: {candidate_goals.shape}")
    logger.info(f"    current_pos: {current_pos.shape}")
    
    predictions, goal_probs = model(
        env_map=env_map,
        history=history,
        candidate_goals=candidate_goals,
        current_pos=current_pos,
        teacher_forcing_ratio=0.5,
        ground_truth=ground_truth
    )
    
    logger.info(f"\n  输出形状:")
    logger.info(f"    predictions: {predictions.shape}")
    logger.info(f"    goal_probs: {goal_probs.shape}")
    
    logger.info("\n✅ 模型测试成功！")


if __name__ == '__main__':
    test_model()
