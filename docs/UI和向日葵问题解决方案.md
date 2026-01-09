# UI 和向日葵问题解决方案

## 问题 1: UI 无法通过 SSH 运行

### 问题原因
- PyQt5 已正确安装（版本 5.15.14）
- **但你通过 SSH 连接服务器，没有图形显示环境**
- 错误：`could not connect to display :0` 和 `Authorization required`

### 解决方案

**你必须在服务器的图形桌面环境中运行 UI，而不是通过 SSH**

#### 方法 A: 在服务器物理显示器前运行
1. 直接在服务器前坐下
2. 打开终端
3. 运行：
   ```bash
   cd /home/zmc/文档/programwork
   conda activate torch-sm120
   python ui/terratnt_professional_ui.py
   ```

#### 方法 B: 使用向日葵远程桌面（推荐）
1. **先在服务器物理屏幕前**打开向日葵应用，获取识别码
2. 在你的电脑上安装向日葵客户端
3. 输入识别码连接到服务器桌面
4. **在向日葵远程桌面中**打开终端运行 UI

#### 方法 C: 配置 VNC 远程桌面
如果无法物理接触服务器，可以配置 VNC：
```bash
sudo apt install tigervnc-standalone-server
vncserver :1 -geometry 1920x1080 -depth 24
# 设置密码后，使用 VNC Viewer 连接到 服务器IP:5901
```

### 为什么 SSH 不行？

SSH 是**命令行连接**，没有图形显示能力。PyQt5 需要：
- X11 显示服务器（`:0` 或 `:1` 等）
- 图形桌面环境（GNOME、KDE 等）
- 显示权限

通过 SSH 连接时，这些都不存在，所以 UI 无法启动。

---

## 问题 2: 向日葵已修复

### 已完成的操作
✅ 安装了 equivs 工具
✅ 创建了 libgconf-2-4 伪装包（版本 3.2.6-99fake）
✅ 成功安装伪装包
✅ 向日葵后台服务正在运行（PID 1665445）

### 如何启动向日葵图形界面

**你需要在服务器的图形桌面环境中启动**（不是 SSH）：

1. **在服务器物理屏幕前**，或通过其他远程桌面（如 VNC）
2. 打开应用程序菜单，搜索"向日葵"或"Sunlogin"
3. 点击启动
4. 记下识别码和验证码

或者命令行启动（在图形环境中）：
```bash
/opt/apps/com.oray.sunlogin.client/files/bin/sunloginclient
```

### 验证安装
```bash
# 检查伪装包
dpkg -l | grep libgconf-2-4
# 输出：ii  libgconf-2-4  3.2.6-99fake  all  Fake package to satisfy sunlogin dependency

# 检查服务状态
systemctl status runsunloginclient.service
# 输出：Active: active (running)
```

---

## 总结

### 核心问题
**你通过 SSH 连接服务器，无法运行图形界面程序（UI 和向日葵）**

### 解决路径
1. **先在服务器物理屏幕前**打开向日葵，获取识别码
2. 在你的电脑上用向日葵客户端连接到服务器
3. **在向日葵远程桌面中**运行 UI：
   ```bash
   cd /home/zmc/文档/programwork
   conda activate torch-sm120
   python ui/terratnt_professional_ui.py
   ```

### 如果无法物理接触服务器
需要先配置 VNC 或其他远程桌面方案，我可以帮你配置。

---

## PyQt5 依赖已安装

所有必要的依赖都已安装：
- ✅ PyQt5 (5.15.14)
- ✅ PyQt5-sip
- ✅ PyQt5-Qt5
- ✅ libxcb-xinerama0
- ✅ libxcb-cursor0
- ✅ 其他 xcb 库

**UI 代码本身没有问题，只是需要在图形环境中运行。**
