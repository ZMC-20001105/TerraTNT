"""
模型评估器 - 统一评估所有模型并生成对比报告
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import MetricsTracker, compute_all_metrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """统一模型评估器"""
    
    def __init__(self, test_loader: DataLoader, device: str = 'cuda'):
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    @torch.no_grad()
    def evaluate_model(
        self, 
        model: torch.nn.Module, 
        model_name: str,
        num_samples: int = 20
    ) -> Dict[str, float]:
        """
        评估单个模型
        
        Args:
            model: 模型实例
            model_name: 模型名称
            num_samples: 多模态采样数量
        
        Returns:
            评估指标字典
        """
        model.to(self.device)
        model.eval()
        
        tracker = MetricsTracker()
        inference_times = []
        
        import time
        
        for batch in self.test_loader:
            history = batch['history'].to(self.device)
            future = batch['future'].to(self.device)
            env_map = batch['env_map'].to(self.device)
            
            # 计时
            start_time = time.time()
            
            # 多模态预测
            if hasattr(model, 'predict'):
                pred = model.predict(history, env_map, num_samples=num_samples)
            else:
                pred = model(history, env_map)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time / history.size(0))  # per sample
            
            # 更新指标
            tracker.update(pred, future)
        
        metrics = tracker.compute()
        metrics['inference_time_ms'] = sum(inference_times) / len(inference_times)
        
        self.results[model_name] = metrics
        
        logger.info(f"{model_name}: ADE={metrics['ade']:.2f}m, FDE={metrics['fde']:.2f}m, "
                   f"GoalAcc={metrics['goal_accuracy']*100:.1f}%, Time={metrics['inference_time_ms']:.1f}ms")
        
        return metrics
    
    def evaluate_all(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict]:
        """评估所有模型"""
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            self.evaluate_model(model, name)
        
        return self.results
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """生成评估报告"""
        report_lines = [
            "=" * 80,
            "MODEL COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"{'Model':<20} {'ADE (m)':<12} {'FDE (m)':<12} {'Goal Acc':<12} {'Time (ms)':<12}",
            "-" * 80
        ]
        
        # 按ADE排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['ade'])
        
        for model_name, metrics in sorted_results:
            line = f"{model_name:<20} {metrics['ade']:<12.2f} {metrics['fde']:<12.2f} " \
                   f"{metrics['goal_accuracy']*100:<12.1f} {metrics['inference_time_ms']:<12.1f}"
            report_lines.append(line)
        
        report_lines.extend([
            "-" * 80,
            "",
            "Best Model by ADE: " + sorted_results[0][0],
            "Best Model by FDE: " + min(self.results.items(), key=lambda x: x[1]['fde'])[0],
            "",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            
            # 同时保存JSON格式
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        return report
    
    def generate_latex_table(self) -> str:
        """生成LaTeX格式的对比表格"""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Model Performance Comparison}",
            r"\label{tab:model_comparison}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & ADE (m) & FDE (m) & Goal Acc (\%) & Time (ms) \\",
            r"\midrule"
        ]
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['ade'])
        
        for i, (model_name, metrics) in enumerate(sorted_results):
            # 最佳结果加粗
            ade_str = f"\\textbf{{{metrics['ade']:.2f}}}" if i == 0 else f"{metrics['ade']:.2f}"
            fde_str = f"\\textbf{{{metrics['fde']:.2f}}}" if metrics['fde'] == min(m['fde'] for m in self.results.values()) else f"{metrics['fde']:.2f}"
            
            line = f"{model_name} & {ade_str} & {fde_str} & {metrics['goal_accuracy']*100:.1f} & {metrics['inference_time_ms']:.1f} \\\\"
            lines.append(line)
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)


def run_full_evaluation(
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str
):
    """
    运行完整的模型评估
    
    Args:
        data_dir: 数据目录
        checkpoint_dir: 模型检查点目录
        output_dir: 输出目录
    """
    from utils.data_processing.trajectory_preprocessor import create_data_loaders
    from models.baselines import YNet, PECNet, TrajectronPP, SocialLSTM, ConstantVelocity
    from models.terratnt import TerraTNT
    
    # 创建测试数据加载器
    _, _, test_loader = create_data_loaders(data_dir, batch_size=32)
    
    # 加载所有模型
    models = {}
    checkpoint_dir = Path(checkpoint_dir)
    
    model_classes = {
        'TerraTNT': TerraTNT,
        'YNet': YNet,
        'PECNet': PECNet,
        'Trajectron++': TrajectronPP,
        'Social-LSTM': SocialLSTM,
        'CV': ConstantVelocity
    }
    
    config = {
        'history_length': 10,
        'future_length': 60,
        'in_channels': 18,
        'hidden_dim': 256
    }
    
    for name, model_class in model_classes.items():
        model = model_class(config)
        
        # 尝试加载检查点
        ckpt_path = checkpoint_dir / f"{name}" / "checkpoint_best.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint for {name}")
        else:
            logger.warning(f"No checkpoint found for {name}, using random weights")
        
        models[name] = model
    
    # 评估
    evaluator = ModelEvaluator(test_loader)
    evaluator.evaluate_all(models)
    
    # 生成报告
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = evaluator.generate_report(output_dir / 'evaluation_report.txt')
    print(report)
    
    latex = evaluator.generate_latex_table()
    with open(output_dir / 'comparison_table.tex', 'w') as f:
        f.write(latex)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    run_full_evaluation(
        data_dir='/home/zmc/文档/programwork/data/processed/synthetic_trajectories/scottish_highlands',
        checkpoint_dir='/home/zmc/文档/programwork/runs',
        output_dir='/home/zmc/文档/programwork/evaluation/results'
    )
