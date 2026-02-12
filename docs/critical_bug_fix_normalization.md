# 关键 Bug 修复：归一化错误导致直线预测

## 🔴 问题发现

### 现象
尽管训练指标显著改善（ADE -39.9%, FDE -59.9%），但可视化显示**预测轨迹依然是直线**，无法跟随 GT 的曲线形状。

### 直线度指标
```
GT mean_norm_dev: 0.0735 | straight_frac(<thr): 0.400  (40% 是直线)
pred mean_norm_dev: 0.0221 | straight_frac(<thr): 1.000  (100% 是直线！)
```

---

## 🐛 根本原因

### 错误的归一化代码 (models/terratnt.py:446-471)

```python
# ❌ 错误：使用段长度进行归一化
s, e = seg_bounds[seg_id]
denom = float(max(1, e - s))  # 段长度，例如 90 个时间步

start_wp_norm = start_wp / denom  # 除以 90
end_wp_norm = end_wp / denom
goal_vector_norm = goal_vector / denom
pos_norm = pos_running / denom
```

### 问题分析

**段长度 vs 空间尺度的混淆**：
- `denom = e - s` 是**时间步数量**（例如 90 步）
- 但 `start_wp`, `goal_vector`, `pos_running` 是**空间坐标**（单位：km）
- **用时间步数量除以空间坐标是完全错误的！**

**具体影响**：
```python
# 假设场景
seg_length = 90  # 时间步
start_wp = [10.0, 20.0]  # km
goal_vector = [20.0, 20.0]  # km
pos_running = [30.0, 40.0]  # km

# 错误归一化
start_wp_norm = [10/90, 20/90] = [0.11, 0.22]
goal_vector_norm = [20/90, 20/90] = [0.22, 0.22]
pos_norm = [30/90, 40/90] = [0.33, 0.44]

# 问题：
# 1. 数值变得极小，失去空间意义
# 2. 不同段长度会导致相同位置有不同的归一化值
# 3. 模型无法学习真实的空间关系和曲线形状
```

---

## ✅ 修复方案

### 正确的归一化代码

```python
# ✅ 正确：使用空间尺度进行归一化
s, e = seg_bounds[seg_id]
seg_len = float(max(1, e - s))  # 段长度（仅用于计算进度）
prog = float(t - s) / seg_len  # 段内进度 [0, 1]

# 使用空间尺度 (goal_norm_denom = 70km)
norm_scale = self.goal_norm_denom  # 70.0 km

start_wp_norm = start_wp / norm_scale
end_wp_norm = end_wp / norm_scale
goal_vector_norm = goal_vector / norm_scale
pos_norm = pos_running / norm_scale
```

### 修复效果

```python
# 正确归一化
norm_scale = 70.0  # km
start_wp_norm = [10/70, 20/70] = [0.14, 0.29]
goal_vector_norm = [20/70, 20/70] = [0.29, 0.29]
pos_norm = [30/70, 40/70] = [0.43, 0.57]

# 优点：
# 1. 数值在合理范围内 [0, 1]
# 2. 保持空间意义和相对关系
# 3. 不同段长度不影响位置的归一化值
# 4. 模型可以学习真实的空间关系和曲线形状
```

---

## 🎯 为什么之前 ADE/FDE 改善了但轨迹还是直线？

### 损失函数修复的效果
```python
loss = (
    1.0 * loss_traj +      # Delta MSE
    10.0 * loss_ade +      # 路径平均误差
    50.0 * loss_fde +      # 终点误差（最重要）
    0.1 * loss_cls +       # 分类损失
    20.0 * loss_wp +       # Waypoint 监督
    1.0 * loss_curv        # 曲率一致性
)
```

**效果**：
- ✅ FDE 权重 50.0 → 终点位置更准确（FDE 改善 59.9%）
- ✅ ADE 权重 10.0 → 整体路径误差降低（ADE 改善 39.9%）

**但是**：
- ❌ 归一化错误 → 模型无法学习曲线形状
- ❌ 只能学到"朝目标方向走"
- ❌ 结果：终点准确但路径是直线

### 形象比喻

**修复前**：
- 模型就像一个被蒙住眼睛的人
- 损失函数告诉他"你要到那个地方"
- 但归一化错误让他看不清空间关系
- 所以他只能"直直地走向目标"

**修复后**：
- 模型可以"看清"空间关系
- 知道 waypoint 的位置、当前位置、目标方向
- 可以学习沿着合理的曲线路径前进

---

## 📊 预期改善

### 修复前（归一化错误）
- **ADE**: 2886 m（改善 39.9%）
- **FDE**: 3251 m（改善 59.9%）
- **轨迹形状**: 100% 直线
- **曲线贴合度**: 极差

### 修复后（归一化正确）
**预期指标**：
- **ADE**: 1500-2500 m（改善 50-70%）
- **FDE**: 2000-3000 m（改善 60-75%）
- **轨迹形状**: 能够生成曲线
- **曲线贴合度**: 显著提升

**预期可视化效果**：
- 预测轨迹能够跟随 GT 的弯曲路径
- 不再是简单的直线
- Waypoint 引导生效
- 环境特征影响轨迹形状

---

## 🔍 教训总结

### 1. 归一化的重要性
- 归一化必须使用**相同量纲**的尺度
- 空间坐标 → 用空间尺度归一化（如 70km）
- 时间步数 → 用时间尺度归一化（如总步数）
- **绝对不能混用！**

### 2. 指标 vs 可视化
- 指标改善不等于问题解决
- **必须结合可视化验证**
- ADE/FDE 可能因为终点准确而改善
- 但轨迹形状可能依然有问题

### 3. 调试方法
- 当指标改善但效果不符合预期时
- 检查数据预处理和归一化
- 检查特征工程的正确性
- 使用可视化发现隐藏问题

---

## 📝 修改文件

- **文件**: `models/terratnt.py`
- **位置**: PaperHierarchicalTrajectoryDecoder.forward() 方法
- **行数**: 445-472
- **修改**: 将 `denom = float(max(1, e - s))` 改为 `norm_scale = self.goal_norm_denom`

---

## 🚀 下一步

1. ✅ 归一化 bug 已修复
2. 🔄 正在重新训练（batch_size=64, 30 epochs）
3. ⏳ 等待训练完成
4. 📊 运行可视化验证曲线预测
5. 📈 分析最终结果

**预计完成时间**: 30-45 分钟

**预期效果**: 预测轨迹能够生成曲线，显著改善曲线贴合度！
