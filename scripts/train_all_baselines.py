"""
批量训练所有基线模型的脚本
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.baseline_trainer import train_baseline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    data_dir = '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/scottish_highlands'
    
    # 所有要训练的模型
    models = ['cv', 'social_lstm', 'ynet', 'pecnet', 'trajectron', 'terratnt']
    
    # 通用配置
    config = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'early_stop_patience': 15,
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
            result = train_baseline(model_name, data_dir, config)
            results[model_name] = result
            logger.info(f"✓ {model_name} training completed: {result}")
        except Exception as e:
            logger.error(f"✗ {model_name} training failed: {e}")
            results[model_name] = {'error': str(e)}
    
    # 打印汇总
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for model, result in results.items():
        if 'error' in result:
            logger.info(f"{model}: FAILED - {result['error']}")
        else:
            logger.info(f"{model}: Best Val Loss = {result.get('best_val_loss', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
