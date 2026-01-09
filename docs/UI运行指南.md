# TerraTNT UI 运行指南

## UI 文件位置
- **主程序**: `/home/zmc/文档/programwork/ui/terratnt_professional_ui.py`
- **依赖环境**: `torch-sm120` conda 环境

## 运行方法

### 方法 1: SSH X11 转发（推荐）

**适用场景**: 从 Windows/Mac/Linux 远程连接

**步骤**:
1. 在本地电脑安装 **MobaXterm** (Windows) 或启用 X11 转发
   - MobaXterm 下载: https://mobaxterm.mobatek.net/
   
2. 用 MobaXterm 连接服务器（自动启用 X11）

3. 在终端运行:
   ```bash
   cd /home/zmc/文档/programwork
   conda activate torch-sm120
   python ui/terratnt_professional_ui.py
   ```

4. UI 窗口会显示在本地电脑上

### 方法 2: 向日葵远程桌面

**适用场景**: 需要完整的远程桌面体验

**步骤**:
1. 在服务器物理屏幕前打开向日葵，获取识别码
2. 在本地电脑安装向日葵客户端
3. 输入识别码连接到服务器桌面
4. 在远程桌面中打开终端运行 UI

### 方法 3: VNC 远程桌面

**适用场景**: 需要持久的远程桌面会话

**安装 VNC Server**:
```bash
sudo apt update
sudo apt install tigervnc-standalone-server tigervnc-common
vncserver :1 -geometry 1920x1080 -depth 24
```

**连接**:
- 使用 VNC Viewer 连接到 `服务器IP:5901`
- 在 VNC 桌面中运行 UI

## UI 功能说明

### 左侧功能区（标签页）
1. **卫星星座**: 配置卫星参数，查看 DEM/LULC 地图
2. **数据加载**: 加载轨迹数据集，显示数据概览
3. **模型训练**: 配置训练参数（当前为演示）
4. **轨迹预测**: 加载 .pkl 轨迹文件，显示真实轨迹

### 右侧可视化区
- 动态显示选中的数据/轨迹
- 支持真实 DEM/LULC 地图渲染
- 轨迹路径可视化（起点、终点、路径）

## 数据加载示例

### 查看真实 DEM 地形
1. 切换到"数据加载"标签页
2. 选择区域（苏格兰高地/波西米亚森林）
3. 点击"显示DEM"按钮
4. 右侧显示真实地形图

### 查看真实轨迹
1. 切换到"轨迹预测"标签页
2. 点击"浏览..."选择 .pkl 文件
   - 路径示例: `/home/zmc/文档/programwork/data/processed/synthetic_trajectories/scottish_highlands/traj_000000_intent1_type1.pkl`
3. 右侧显示轨迹路径

## 依赖检查

如果 UI 无法启动，检查以下依赖:

```bash
conda activate torch-sm120
pip install PyQt5 matplotlib numpy
```

如果需要显示 DEM/LULC:
```bash
pip install rasterio scipy
```

## 常见问题

### Q: 提示 "Could not load Qt platform plugin"
**A**: 这是 SSH 环境问题，需要使用 X11 转发或远程桌面

### Q: 中文显示为方块
**A**: UI 已自动检测系统中文字体，如果仍有问题，安装:
```bash
sudo apt install fonts-noto-cjk
```

### Q: UI 启动后无响应
**A**: 检查 DISPLAY 环境变量:
```bash
echo $DISPLAY
# 应该显示 :0 或 localhost:10.0 等
```

## 截图保存

如果需要保存 UI 截图（在有图形环境时）:
```python
# 在 UI 中按 Ctrl+S 或使用系统截图工具
```
