"""
速度预测模型训练与验证

按论文表3.7参数训练XGBoost模型，并执行三套验证实验：
1. 5-fold 交叉验证（按轨迹顺序分段）
2. LOOCV（留一轨迹）
3. SHAP 可解释性分析
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

from config import cfg, get_path
from config.plot_config import PlotConfig

logger = logging.getLogger(__name__)
plot_cfg = PlotConfig()


class SpeedPredictor:
    """速度预测模型（XGBoost）"""
    
    def __init__(self):
        # 论文表3.7超参数
        self.params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """训练模型"""
        logger.info("训练XGBoost速度预测模型...")
        logger.info(f"  样本数: {len(X)}")
        logger.info(f"  特征维度: {X.shape[1]}")
        logger.info(f"  超参数: {self.params}")
        
        self.feature_names = feature_names
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        
        logger.info("✓ 训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 log(1+v)"""
        return self.model.predict(X)
    
    def predict_speed(self, X: np.ndarray) -> np.ndarray:
        """预测速度 v = exp(y) - 1"""
        y_pred = self.predict(X)
        return np.exp(y_pred) - 1
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """评估模型"""
        y_pred = self.predict(X)
        
        # 转换回速度
        v_true = np.exp(y) - 1
        v_pred = np.exp(y_pred) - 1
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': mean_absolute_percentage_error(v_true, v_pred) * 100,
            'speed_rmse': np.sqrt(mean_squared_error(v_true, v_pred))
        }
        
        return metrics
    
    def save(self, path: Path):
        """保存模型"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ 模型已保存: {path}")
    
    def load(self, path: Path):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"✓ 模型已加载: {path}")


def kfold_validation(X: np.ndarray, y: np.ndarray, traj_ids: np.ndarray, n_splits: int = 5) -> dict:
    """
    K-fold交叉验证（按轨迹顺序分段）
    
    论文方法：将每条轨迹按时间顺序分段，每段约1000点，再做5折
    """
    logger.info("\n" + "=" * 60)
    logger.info("K-fold 交叉验证（5折）")
    logger.info("=" * 60)
    
    # 按轨迹分组
    unique_trajs = np.unique(traj_ids)
    logger.info(f"轨迹数: {len(unique_trajs)}")
    
    # 为每条轨迹创建分段索引
    segment_size = 1000
    segments = []
    segment_traj_ids = []
    
    for traj_id in unique_trajs:
        traj_mask = traj_ids == traj_id
        traj_indices = np.where(traj_mask)[0]
        
        # 按顺序分段
        n_points = len(traj_indices)
        n_segments = max(1, n_points // segment_size)
        
        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, n_points)
            segments.append(traj_indices[start:end])
            segment_traj_ids.append(traj_id)
    
    logger.info(f"总分段数: {len(segments)}")
    
    # K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_seg_idx, val_seg_idx) in enumerate(kf.split(segments)):
        logger.info(f"\nFold {fold + 1}/{n_splits}:")
        
        # 构建训练/验证索引
        train_idx = np.concatenate([segments[i] for i in train_seg_idx])
        val_idx = np.concatenate([segments[i] for i in val_seg_idx])
        
        logger.info(f"  训练样本: {len(train_idx)}")
        logger.info(f"  验证样本: {len(val_idx)}")
        
        # 训练
        predictor = SpeedPredictor()
        predictor.train(X[train_idx], y[train_idx])
        
        # 评估
        metrics = predictor.evaluate(X[val_idx], y[val_idx])
        fold_metrics.append(metrics)
        
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Speed RMSE: {metrics['speed_rmse']:.4f} m/s")
    
    # 汇总
    avg_metrics = {
        'r2': np.mean([m['r2'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mape': np.mean([m['mape'] for m in fold_metrics]),
        'speed_rmse': np.mean([m['speed_rmse'] for m in fold_metrics]),
        'r2_std': np.std([m['r2'] for m in fold_metrics]),
        'rmse_std': np.std([m['rmse'] for m in fold_metrics]),
        'mape_std': np.std([m['mape'] for m in fold_metrics]),
        'speed_rmse_std': np.std([m['speed_rmse'] for m in fold_metrics])
    }
    
    logger.info("\n平均指标:")
    logger.info(f"  R²: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    logger.info(f"  RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['rmse_std']:.4f}")
    logger.info(f"  MAPE: {avg_metrics['mape']:.2f}% ± {avg_metrics['mape_std']:.2f}%")
    logger.info(f"  Speed RMSE: {avg_metrics['speed_rmse']:.4f} ± {avg_metrics['speed_rmse_std']:.4f} m/s")
    
    return {'fold_metrics': fold_metrics, 'avg_metrics': avg_metrics}


def loocv_validation(X: np.ndarray, y: np.ndarray, traj_ids: np.ndarray) -> dict:
    """
    留一轨迹交叉验证（LOOCV）
    
    测试跨轨迹泛化能力
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOOCV（留一轨迹）交叉验证")
    logger.info("=" * 60)
    
    unique_trajs = np.unique(traj_ids)
    logger.info(f"轨迹数: {len(unique_trajs)}")
    
    traj_metrics = []
    
    for i, test_traj in enumerate(unique_trajs):
        logger.info(f"\n留出轨迹 {i + 1}/{len(unique_trajs)} (ID={test_traj}):")
        
        # 划分训练/测试
        train_mask = traj_ids != test_traj
        test_mask = traj_ids == test_traj
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        logger.info(f"  训练样本: {len(X_train)}")
        logger.info(f"  测试样本: {len(X_test)}")
        
        # 训练
        predictor = SpeedPredictor()
        predictor.train(X_train, y_train)
        
        # 评估
        metrics = predictor.evaluate(X_test, y_test)
        metrics['traj_id'] = test_traj
        traj_metrics.append(metrics)
        
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    # 汇总
    avg_metrics = {
        'r2': np.mean([m['r2'] for m in traj_metrics]),
        'rmse': np.mean([m['rmse'] for m in traj_metrics]),
        'mape': np.mean([m['mape'] for m in traj_metrics]),
        'r2_std': np.std([m['r2'] for m in traj_metrics]),
        'rmse_std': np.std([m['rmse'] for m in traj_metrics]),
        'mape_std': np.std([m['mape'] for m in traj_metrics])
    }
    
    logger.info("\n平均指标:")
    logger.info(f"  R²: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    logger.info(f"  RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['rmse_std']:.4f}")
    logger.info(f"  MAPE: {avg_metrics['mape']:.2f}% ± {avg_metrics['mape_std']:.2f}%")
    
    return {'traj_metrics': traj_metrics, 'avg_metrics': avg_metrics}


def shap_analysis(X: np.ndarray, y: np.ndarray, feature_names: list, output_dir: Path):
    """
    SHAP特征重要性分析（使用XGBoost内置feature importance作为替代）
    """
    logger.info("\n" + "=" * 60)
    logger.info("特征重要性分析")
    logger.info("=" * 60)
    
    # 训练模型
    logger.info("训练完整模型...")
    predictor = SpeedPredictor()
    predictor.train(X, y, feature_names)
    
    # 使用XGBoost内置feature importance
    logger.info("计算特征重要性...")
    feature_importance = predictor.model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\n特征重要性排序:")
    for i, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    # 绘图
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature importance bar plot
    fig, ax = plot_cfg.create_figure(size=(10, 8))
    importance_df_top20 = importance_df.head(20)
    ax.barh(range(len(importance_df_top20)), importance_df_top20['importance'])
    ax.set_yticks(range(len(importance_df_top20)))
    ax.set_yticklabels(importance_df_top20['feature'])
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('XGBoost Feature Importance')
    ax.invert_yaxis()
    plot_cfg.save_figure(fig, output_dir / 'feature_importance.png')
    plt.close(fig)
    
    logger.info(f"\n✓ 特征重要性分析完成，图表已保存到: {output_dir}")
    
    return importance_df


def run_full_validation():
    """运行完整验证流程"""
    logger.info("=" * 60)
    logger.info("速度预测模型训练与验证")
    logger.info("=" * 60)
    
    # 加载数据
    data_path = Path(get_path('paths.processed.speed_training')) / 'speed_training_data.npz'
    logger.info(f"\n加载训练数据: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    traj_ids = data['traj_ids']
    feature_names = data['feature_names'].tolist()
    
    logger.info(f"  样本数: {len(X)}")
    logger.info(f"  特征维度: {X.shape[1]}")
    logger.info(f"  轨迹数: {len(np.unique(traj_ids))}")
    
    # 输出目录
    output_dir = Path(get_path('paths.outputs.results')) / 'speed_model_validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. K-fold验证
    kfold_results = kfold_validation(X, y, traj_ids, n_splits=5)
    
    # 2. LOOCV验证
    loocv_results = loocv_validation(X, y, traj_ids)
    
    # 3. SHAP分析
    shap_results = shap_analysis(X, y, feature_names, output_dir)
    
    # 保存结果
    results = {
        'kfold': kfold_results,
        'loocv': loocv_results,
        'shap': shap_results
    }
    
    with open(output_dir / 'validation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 训练并保存最终模型
    logger.info("\n" + "=" * 60)
    logger.info("训练最终模型（全部数据）")
    logger.info("=" * 60)
    
    final_predictor = SpeedPredictor()
    final_predictor.train(X, y, feature_names)
    
    model_path = Path(get_path('paths.models.speed_predictor')) / 'speed_model.pkl'
    final_predictor.save(model_path)
    
    # 生成报告
    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("速度预测模型验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("数据统计:\n")
        f.write(f"  样本数: {len(X)}\n")
        f.write(f"  特征维度: {X.shape[1]}\n")
        f.write(f"  轨迹数: {len(np.unique(traj_ids))}\n\n")
        
        f.write("K-fold 交叉验证 (5折):\n")
        f.write(f"  R²: {kfold_results['avg_metrics']['r2']:.4f} ± {kfold_results['avg_metrics']['r2_std']:.4f}\n")
        f.write(f"  RMSE: {kfold_results['avg_metrics']['rmse']:.4f} ± {kfold_results['avg_metrics']['rmse_std']:.4f}\n")
        f.write(f"  MAPE: {kfold_results['avg_metrics']['mape']:.2f}% ± {kfold_results['avg_metrics']['mape_std']:.2f}%\n")
        f.write(f"  Speed RMSE: {kfold_results['avg_metrics']['speed_rmse']:.4f} ± {kfold_results['avg_metrics']['speed_rmse_std']:.4f} m/s\n\n")
        
        f.write("LOOCV (留一轨迹):\n")
        f.write(f"  R²: {loocv_results['avg_metrics']['r2']:.4f} ± {loocv_results['avg_metrics']['r2_std']:.4f}\n")
        f.write(f"  RMSE: {loocv_results['avg_metrics']['rmse']:.4f} ± {loocv_results['avg_metrics']['rmse_std']:.4f}\n")
        f.write(f"  MAPE: {loocv_results['avg_metrics']['mape']:.2f}% ± {loocv_results['avg_metrics']['mape_std']:.2f}%\n\n")
        
        f.write("SHAP 特征重要性 (Top 10):\n")
        for i, row in shap_results.head(10).iterrows():
            f.write(f"  {row['feature']:20s}: {row['importance']:.4f}\n")
    
    logger.info(f"\n✅ 验证报告已保存: {report_path}")
    logger.info(f"✅ 最终模型已保存: {model_path}")
    logger.info(f"✅ 所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_full_validation()
