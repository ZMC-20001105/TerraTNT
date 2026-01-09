# VNC 远程桌面配置指南

## 快速启动 VNC（已配置完成）

VNC Server 已安装并配置完成。现在启动 VNC：

### 步骤 1: 启动 VNC Server

在 SSH 终端中运行：
```bash
vncserver :1 -geometry 1920x1080 -depth 24
```

**首次运行会要求设置密码**：
- 输入 VNC 连接密码（6-8位）
- 确认密码
- 是否设置仅查看密码？选择 `n`

### 步骤 2: 在本地电脑连接 VNC

1. **下载 VNC Viewer**（免费）：
   - Windows/Mac: https://www.realvnc.com/en/connect/download/viewer/
   - 或使用 TigerVNC Viewer

2. **连接到服务器**：
   - 地址：`服务器IP:5901`（例如 `192.168.1.100:5901`）
   - 输入刚才设置的密码

3. **连接成功后**，你会看到服务器的图形桌面

### 步骤 3: 在 VNC 桌面中运行 UI

在 VNC 远程桌面中打开终端，运行：
```bash
cd /home/zmc/文档/programwork
conda activate torch-sm120
python ui/terratnt_professional_ui.py
```

### 步骤 4: 在 VNC 桌面中启动向日葵

在 VNC 远程桌面中：
1. 打开应用程序菜单
2. 搜索"向日葵"或"Sunlogin"
3. 点击启动
4. 记下识别码

然后你就可以用向日葵代替 VNC（向日葵性能更好）

---

## VNC 管理命令

### 查看运行中的 VNC 会话
```bash
vncserver -list
```

### 停止 VNC 会话
```bash
vncserver -kill :1
```

### 重启 VNC 会话
```bash
vncserver -kill :1
vncserver :1 -geometry 1920x1080 -depth 24
```

---

## 常见问题

### Q: VNC 连接后显示灰屏或黑屏
**A**: 桌面环境未正确启动，检查 `~/.vnc/xstartup` 文件

### Q: VNC 连接很慢
**A**: 
1. 降低分辨率：`vncserver :1 -geometry 1280x720 -depth 16`
2. 或使用向日葵（性能更好）

### Q: 忘记 VNC 密码
**A**: 删除密码文件重新设置：
```bash
rm ~/.vnc/passwd
vncserver :1
```

---

## 端口说明

- `:1` 对应端口 `5901`
- `:2` 对应端口 `5902`
- 以此类推

**防火墙设置**（如果需要）：
```bash
sudo ufw allow 5901/tcp
```

---

## 推荐流程

1. ✅ 启动 VNC Server（已配置）
2. ✅ 用 VNC Viewer 连接到服务器桌面
3. ✅ 在 VNC 桌面中启动向日葵，获取识别码
4. ✅ 之后用向日葵连接（性能更好，功能更多）
5. ✅ 在远程桌面中运行 UI

这样你就可以完全远程操作服务器的图形界面了。
