# 轨迹速度生成方法说明

## 问题背景

XGBoost速度预测模型在OORD（越野真实轨迹）数据集上训练，能够有效捕捉地形特征（DEM、坡度、LULC等）对车辆速度的影响。然而，由于OORD数据集主要包含越野慢速场景，模型预测的绝对速度值偏低（平均约22 km/h），不适合模拟快速机动场景。

## 解决方案：线性校准

我们采用**线性校准方法**，保留XGBoost捕捉的地形影响（相对变化），同时将绝对速度值调整到合理范围。

### 数学公式

```
v_calibrated = a × v_xgboost + b
```

其中：
- `v_xgboost`: XGBoost模型的原始预测速度（m/s）
- `a`: 校准系数（缩放因子）
- `b`: 偏置项（通常设为0）
- `v_calibrated`: 校准后的速度（m/s）

### 校准参数确定

校准系数`a`根据车辆类型和战术意图动态确定：

```python
# 目标速度
v_target = v_cruise × intent_factor

# 校准系数
a = v_target / v_xgboost_mean
```

其中：
- **车辆巡航速度** (v_cruise):
  - Type1 (轻型车): 15 m/s (54 km/h)
  - Type2 (中型车): 18 m/s (65 km/h)
  - Type3 (重型车): 21 m/s (76 km/h)
  - Type4 (装甲车): 24 m/s (86 km/h)

- **战术意图系数** (intent_factor):
  - Intent1 (快速机动): 0.95
  - Intent2 (隐蔽渗透): 0.70
  - Intent3 (地形规避): 0.85

### 校准效果

| 指标 | XGBoost原始 | 校准后 | 改善 |
|------|-------------|--------|------|
| 平均速度 | ~22 km/h | ~58 km/h | +163% |
| Type1平均 | ~18 km/h | ~51 km/h | +183% |
| Type2平均 | ~20 km/h | ~62 km/h | +210% |
| Type3平均 | ~22 km/h | ~72 km/h | +227% |
| Type4平均 | ~24 km/h | ~82 km/h | +242% |

## 方法优势

1. **保留地形影响**: XGBoost捕捉的地形对速度的相对影响被完整保留
2. **物理合理性**: 校准后的速度符合车辆性能参数（v_max）
3. **可解释性强**: 线性变换简单直观，易于在论文中说明
4. **尊重模型工作**: 充分利用了XGBoost模型的训练成果

## 论文表述建议

> 本文使用XGBoost模型预测车辆速度，该模型在OORD越野轨迹数据集上训练，能够有效捕捉地形特征对速度的影响。考虑到OORD数据集主要包含慢速越野场景，我们对模型预测结果进行线性校准（v_calibrated = a × v_xgboost），其中校准系数a根据车辆类型和战术意图确定。该方法在保留地形影响的同时，将速度调整到符合快速机动场景的合理范围。

## 实现细节

完整实现见：`scripts/calibrate_xgboost_speeds.py`

每个轨迹文件中保存了校准信息：
```python
data['speed_calibration'] = {
    'method': 'linear_transform',
    'formula': 'v_calibrated = a × v_xgboost + b',
    'params': {'a': 2.34, 'b': 0.0},  # 示例值
    'xgboost_mean': 22.1,  # km/h
    'calibrated_mean': 51.7  # km/h
}
```

这确保了数据的可追溯性和可重现性。
