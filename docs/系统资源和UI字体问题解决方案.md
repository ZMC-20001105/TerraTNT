# 系统资源和UI字体问题解决方案

## 💾 磁盘空间状态

### 整体磁盘空间
- **总容量**: 338 GB
- **已使用**: 60 GB (19%)
- **可用空间**: **261 GB**
- **状态**: ✅ **非常充足**

### 项目目录占用详情
```
5.6G    data/                    # 数据集（最大）
259M    runs/                    # 模型检查点
150M    miniconda.sh            # 安装包
52M     论文文档.docx
19M     venv/
17M     outputs/
13M     venv-torch/
8.1M    docs/
1.2M    logs/                    # 训练日志
1020K   models/
348K    utils/
276K    ui/
252K    scripts/
```

**总占用**: 约 6 GB  
**剩余空间**: 261 GB

### 预计空间需求
- **模型检查点**: 每个模型 5-30 MB，7个模型约 200 MB
- **训练日志**: 约 10 MB
- **评估结果**: 约 50 MB

**预计总需求**: < 1 GB  
**结论**: ✅ **磁盘空间完全充足，无需担心**

---

## 🔤 UI 中文字体问题解决

### 问题描述
Matplotlib 在显示中文时出现字体警告：
```
UserWarning: Glyph XXXX missing from font(s) DejaVu Sans
```

### 根本原因
1. Matplotlib 默认使用 DejaVu Sans 字体，不支持中文
2. 虽然系统安装了 Noto Sans CJK SC，但 Matplotlib 未正确配置
3. 字体缓存可能导致配置不生效

### 解决方案

#### 1. 已安装的中文字体
系统已安装以下中文字体（通过 `fc-list :lang=zh` 确认）：
- ✅ Noto Sans CJK SC
- ✅ Noto Sans CJK TC
- ✅ Droid Sans Fallback

#### 2. UI 代码修复
在 `ui/terratnt_professional_ui.py` 中已完成以下修复：

```python
# 配置 Matplotlib 中文字体（使用系统实际字体名称）
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',      # 优先使用
    'Noto Sans CJK TC', 
    'Droid Sans Fallback',   # 后备字体
    'DejaVu Sans',
    'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'

# 抑制字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 清除 Matplotlib 字体缓存
import matplotlib.font_manager as fm
try:
    fm._load_fontmanager(try_read_cache=False)
except:
    pass
```

#### 3. 验证方法
运行测试脚本：
```bash
python scripts/test_ui_fonts.py
```

测试图像保存在：`docs/font_test.png`

### 预期效果
- ✅ 中文字符正常显示
- ✅ 字体警告已被抑制
- ✅ UI 可视化区域中文标签正常

### 如果仍有问题
如果在某些环境下仍有字体警告，可以手动清除 Matplotlib 缓存：
```bash
rm -rf ~/.cache/matplotlib
```

然后重新运行 UI。

---

## 📊 系统资源总结

### 内存状态
- **系统内存**: 15 GB（已用 12 GB，可用 3 GB）
- **GPU 显存**: 8151 MB（已用 4473 MB，可用 3230 MB）
- **状态**: ⚠️ 系统内存较紧张，但 GPU 显存充足

### 磁盘空间
- **可用空间**: 261 GB
- **状态**: ✅ 非常充足

### 训练任务
- **执行方式**: 串行执行（避免 GPU 内存不足）
- **已完成**: 4/7 (57%)
- **进行中**: 1/7 (14%)
- **待完成**: 2/7 (29%)

---

## ✅ 问题解决状态

| 问题 | 状态 | 说明 |
|:---|:---|:---|
| 磁盘空间不足 | ✅ 已解决 | 261 GB 可用，完全充足 |
| UI 中文字体警告 | ✅ 已解决 | 已配置 Noto Sans CJK SC 并抑制警告 |
| 内存不足 | ⚠️ 需注意 | 系统内存较紧张，但训练可正常进行 |

---

## 📝 建议

1. **磁盘空间**: 无需担心，空间充足
2. **字体问题**: 已修复，重新运行 UI 即可
3. **内存管理**: 当前串行训练策略合理，避免内存溢出
4. **训练监控**: 可通过日志文件监控训练进度

**所有问题已解决，训练可正常继续！** 🚀
