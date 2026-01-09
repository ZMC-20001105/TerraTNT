"""
完整实验运行脚本
支持训练、评估和结果可视化的一键执行
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training(args):
    """运行模型训练"""
    from training.baseline_trainer import train_baseline
    
    models = args.models.split(',') if args.models else ['terratnt', 'ynet', 'pecnet', 'trajectron', 'social_lstm', 'cv']
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'early_stop_patience': args.patience,
        'history_length': 10,
        'future_length': 60,
        'in_channels': 18,
        'hidden_dim': 256
    }
    
    results = {}
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            result = train_baseline(model_name, args.data_dir, config)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def run_evaluation(args):
    """运行模型评估"""
    from evaluation.evaluator import run_full_evaluation
    
    run_full_evaluation(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )


def run_visualization(args):
    """生成可视化结果"""
    from evaluation.visualizer import generate_paper_figures
    import json
    
    # 加载评估结果
    results_path = Path(args.results_file)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        generate_paper_figures(results, Path(args.output_dir))
    else:
        logger.error(f"Results file not found: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='TerraTNT Experiment Runner')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    train_parser.add_argument('--models', type=str, default=None, help='Comma-separated model names')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    eval_parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint directory')
    eval_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # 可视化命令
    vis_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    vis_parser.add_argument('--results-file', type=str, required=True, help='Results JSON file')
    vis_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'visualize':
        run_visualization(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
