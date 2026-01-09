#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评估框架 - 评估所有模型在 FAS1/2/3 三个阶段的性能
"""
import os
import sys
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# 添加项目路径
sys.path.append('/home/zmc/文档/programwork')

from models.baselines.social_lstm import SocialLSTM
from models.baselines.ynet import YNet
from models.baselines.pecnet import PECNet
from models.baselines.trajectron import TrajectronPP
from models.terratnt import TerraTNT

class FASTrajectoryDataset(torch.utils.data.Dataset):
    """FAS 阶段特定的轨迹数据集"""
    
    def __init__(self, data_root, fas_config, phase='fas1', 
                 history_len=10, future_len=60, num_candidates=6):
        self.data_root = Path(data_root)
        self.phase = phase
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        
        self.file_list = fas_config[phase]['files']
        
        # 预处理样本
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        for file_name in tqdm(self.file_list, desc=f"加载{self.phase}数据"):
            file_path = self.data_root / file_name
            if not file_path.exists():
                continue
            
            with open(file_path, 'rb') as f:
                traj_data = pickle.load(f)
            
            path = np.array(traj_data['path'])
            
            if len(path) < self.history_len + self.future_len:
                continue
            
            # 提取历史和未来轨迹
            history = path[:self.history_len]
            future = path[self.history_len:self.history_len + self.future_len]
            
            # 生成候选终点
            goal = path[-1]
            candidates = self._generate_candidates(goal, traj_data.get('goal_utm', goal))
            
            self.samples.append({
                'history': history,
                'future': future,
                'candidates': candidates,
                'goal': goal
            })
    
    def _generate_candidates(self, true_goal, goal_utm):
        """生成候选终点（包含真实终点）"""
        candidates = [true_goal]
        
        # 生成随机候选点
        for _ in range(self.num_candidates - 1):
            noise = np.random.randn(2) * 1000  # 1km 标准差
            fake_candidate = true_goal + noise
            candidates.append(fake_candidate)
        
        return np.array(candidates)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'history': torch.FloatTensor(sample['history']),
            'future': torch.FloatTensor(sample['future']),
            'candidates': torch.FloatTensor(sample['candidates']),
            'goal': torch.FloatTensor(sample['goal'])
        }

class ModelEvaluator:
    """统一模型评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # 数据集配置
        self.data_root = '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest'
        self.fas_config_path = '/home/zmc/文档/programwork/data/processed/fas_splits/bohemian_forest/fas_splits.json'
        
        # 加载 FAS 配置
        with open(self.fas_config_path, 'r') as f:
            self.fas_config = json.load(f)
        
        print(f"✓ 加载 FAS 配置: {self.fas_config_path}")
        print(f"  - FAS1 样本数: {self.fas_config['fas1']['num_samples']}")
        print(f"  - FAS2 样本数: {self.fas_config['fas2']['num_samples']}")
        print(f"  - FAS3 样本数: {self.fas_config['fas3']['num_samples']}")
    
    def load_model(self, model_name, model_path, model_config):
        """加载训练好的模型"""
        print(f"\n加载模型: {model_name}")
        print(f"  路径: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"  ✗ 模型文件不存在: {model_path}")
            return None
        
        # 创建模型实例 - 所有基线模型都使用 config 字典
        if model_name in ['social_lstm', 'ynet', 'pecnet', 'trajectron']:
            # 基线模型统一使用 config 字典初始化
            if model_name == 'social_lstm':
                model = SocialLSTM(model_config)
            elif model_name == 'ynet':
                model = YNet(model_config)
            elif model_name == 'pecnet':
                model = PECNet(model_config)
            elif model_name == 'trajectron':
                model = TrajectronPP(model_config)
        elif model_name == 'terratnt':
            model = TerraTNT(
                history_len=model_config['history_len'],
                future_len=model_config['future_len'],
                hidden_dim=model_config.get('hidden_dim', 256),
                num_goals=model_config.get('num_goals', 6),
                map_size=model_config.get('map_size', 128),
                in_channels=model_config.get('in_channels', 18)
            )
        else:
            raise ValueError(f"未知模型类型: {model_name}")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  ✓ 模型加载成功")
        return model
    
    def compute_metrics(self, predictions, ground_truth):
        """计算评估指标"""
        # predictions: (batch, future_len, 2)
        # ground_truth: (batch, future_len, 2)
        
        # ADE (Average Displacement Error)
        displacement = np.linalg.norm(predictions - ground_truth, axis=2)  # (batch, future_len)
        ade = np.mean(displacement)
        
        # FDE (Final Displacement Error)
        fde = np.mean(displacement[:, -1])
        
        # MR (Miss Rate) - 如果最终位置误差 > 2m，则认为失败
        miss_threshold = 2.0  # 2 meters
        mr = np.mean(displacement[:, -1] > miss_threshold)
        
        return {
            'ADE': ade,
            'FDE': fde,
            'MR': mr
        }
    
    def evaluate_model_on_phase(self, model, model_name, phase, batch_size=32):
        """在指定 FAS 阶段评估模型"""
        print(f"\n评估 {model_name} 在 {phase} 阶段...")
        
        # 创建数据集
        dataset = FASTrajectoryDataset(
            data_root=self.data_root,
            fas_config=self.fas_config,
            phase=phase,
            history_len=10,
            future_len=60,
            num_candidates=6
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{model_name} - {phase}"):
                history = batch['history'].to(self.device)
                future = batch['future'].to(self.device)
                candidates = batch['candidates'].to(self.device)
                
                # 加载真实环境地图
                from utils.env_data_loader import get_env_loader
                env_loader = get_env_loader('bohemian_forest')
                
                # 提取当前位置的绝对坐标（假设在相对坐标系统中，需要从数据集获取）
                # 这里暂时使用零填充，实际应该从轨迹数据中获取绝对位置
                # TODO: 修改数据集以包含绝对位置信息
                env_map = torch.zeros(history.size(0), 18, 128, 128).to(self.device)
                print("警告: 评估时环境数据加载需要绝对坐标，当前使用零填充")
                
                # 根据模型类型进行推理
                if model_name == 'terratnt':
                    # TerraTNT 需要环境地图和候选点
                    current_pos = history[:, -1, :]
                    pred, _ = model(env_map, history, candidates, current_pos)
                elif model_name == 'social_lstm':
                    # Social-LSTM 只需要历史轨迹
                    pred = model(history, env_map)
                elif model_name in ['ynet', 'pecnet', 'trajectron']:
                    # 这些模型需要环境地图
                    pred = model(history, env_map)
                else:
                    raise ValueError(f"未知模型: {model_name}")
                
                # 处理输出
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                all_predictions.append(pred.cpu().numpy())
                all_ground_truth.append(future.cpu().numpy())
        
        # 合并所有批次
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        
        # 计算指标
        metrics = self.compute_metrics(predictions, ground_truth)
        
        print(f"  ✓ {phase} 评估完成:")
        print(f"    - ADE: {metrics['ADE']:.4f} m")
        print(f"    - FDE: {metrics['FDE']:.4f} m")
        print(f"    - MR:  {metrics['MR']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self):
        """评估所有模型"""
        
        # 模型配置 - 使用训练时的实际配置
        models_config = {
            'social_lstm': {
                'path': '/home/zmc/文档/programwork/runs/social_lstm/20260108_173130/best_model.pth',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,  # 训练时使用的配置
                    'embedding_dim': 64
                }
            },
            'ynet': {
                'path': '/home/zmc/文档/programwork/runs/ynet/*/best_model.pth',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'encoder_h_dim': 128,
                    'decoder_h_dim': 256,
                    'emb_dim': 32
                }
            },
            'pecnet': {
                'path': '/home/zmc/文档/programwork/runs/pecnet_20260108_181515/best_model.pth',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'enc_latent_size': 64,
                    'dec_size': 256,
                    'predictor_size': 128,
                    'non_local_theta_size': 64,
                    'non_local_phi_size': 64,
                    'non_local_g_size': 32,
                    'fdim': 32,
                    'zdim': 32,
                    'sigma': 1.3
                }
            },
            'trajectron': {
                'path': '/home/zmc/文档/programwork/runs/trajectron/20260108_185847/best_model.pth',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'num_modes': 5
                }
            }
        }
        
        # TerraTNT 三个阶段
        terratnt_configs = {
            'terratnt_fas1': {
                'path': '/home/zmc/文档/programwork/runs/terratnt_fas1/20260108_221129/best_model.pth',
                'phase': 'fas1',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'num_goals': 6,
                    'map_size': 128,
                    'in_channels': 18
                }
            },
            'terratnt_fas2': {
                'path': '/home/zmc/文档/programwork/runs/terratnt_fas2/20260108_234820/best_model.pth',
                'phase': 'fas2',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'num_goals': 6,
                    'map_size': 128,
                    'in_channels': 18
                }
            },
            'terratnt_fas3': {
                'path': '/home/zmc/文档/programwork/runs/terratnt_fas3/20260109_002958/best_model.pth',
                'phase': 'fas3',
                'config': {
                    'history_len': 10,
                    'future_len': 60,
                    'hidden_dim': 256,
                    'num_goals': 6,
                    'map_size': 128,
                    'in_channels': 18
                }
            }
        }
        
        # 查找 YNet 模型路径
        ynet_runs = list(Path('/home/zmc/文档/programwork/runs').glob('ynet/*/best_model.pth'))
        if ynet_runs:
            models_config['ynet']['path'] = str(ynet_runs[0])
        
        results = {}
        
        # 评估基线模型（在所有三个阶段）
        for model_name, config in models_config.items():
            if not os.path.exists(config['path']):
                print(f"\n✗ 跳过 {model_name}，模型文件不存在")
                continue
            
            model = self.load_model(model_name, config['path'], config['config'])
            if model is None:
                continue
            
            results[model_name] = {}
            for phase in ['fas1', 'fas2', 'fas3']:
                metrics = self.evaluate_model_on_phase(model, model_name, phase)
                results[model_name][phase] = metrics
            
            del model
            torch.cuda.empty_cache()
        
        # 评估 TerraTNT（每个阶段使用对应训练的模型）
        for model_name, config in terratnt_configs.items():
            model = self.load_model('terratnt', config['path'], config['config'])
            if model is None:
                continue
            
            phase = config['phase']
            metrics = self.evaluate_model_on_phase(model, 'terratnt', phase)
            
            if 'terratnt' not in results:
                results['terratnt'] = {}
            results['terratnt'][phase] = metrics
            
            del model
            torch.cuda.empty_cache()
        
        self.results = results
        return results
    
    def generate_report(self, output_path='/home/zmc/文档/programwork/evaluation_results.json'):
        """生成评估报告"""
        print(f"\n生成评估报告...")
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        results_native = convert_to_native(self.results)
        
        # 保存 JSON 结果
        with open(output_path, 'w') as f:
            json.dump(results_native, f, indent=2)
        print(f"✓ JSON 结果已保存: {output_path}")
        
        # 生成 Markdown 表格
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 模型评估结果\n\n")
            
            for phase in ['fas1', 'fas2', 'fas3']:
                f.write(f"## {phase.upper()} 阶段\n\n")
                f.write("| 模型 | ADE (m) | FDE (m) | MR |\n")
                f.write("|:---|:---|:---|:---|\n")
                
                for model_name in sorted(self.results.keys()):
                    if phase in self.results[model_name]:
                        metrics = self.results[model_name][phase]
                        f.write(f"| {model_name} | {metrics['ADE']:.4f} | {metrics['FDE']:.4f} | {metrics['MR']:.4f} |\n")
                
                f.write("\n")
        
        print(f"✓ Markdown 报告已保存: {md_path}")
        
        # 打印总结
        print("\n" + "="*60)
        print("评估结果总结")
        print("="*60)
        
        for phase in ['fas1', 'fas2', 'fas3']:
            print(f"\n{phase.upper()} 阶段:")
            print("-" * 60)
            print(f"{'模型':<20} {'ADE (m)':<12} {'FDE (m)':<12} {'MR':<10}")
            print("-" * 60)
            
            for model_name in sorted(self.results.keys()):
                if phase in self.results[model_name]:
                    metrics = self.results[model_name][phase]
                    print(f"{model_name:<20} {metrics['ADE']:<12.4f} {metrics['FDE']:<12.4f} {metrics['MR']:<10.4f}")

def main():
    """主函数"""
    print("="*60)
    print("TerraTNT 统一评估框架")
    print("="*60)
    
    # 创建评估器
    evaluator = ModelEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 评估所有模型
    results = evaluator.evaluate_all_models()
    
    # 生成报告
    evaluator.generate_report()
    
    print("\n✓ 评估完成！")

if __name__ == '__main__':
    main()
