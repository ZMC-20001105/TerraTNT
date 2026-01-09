# 设置 Google Cloud Storage Bucket

## 为什么需要 GCS？

1. ❌ **Drive 导出**：服务账号没有存储配额
2. ❌ **直接下载**：文件太大（超过 50MB 限制）
3. ✅ **GCS 导出**：唯一可行的方案

## 快速设置步骤（5分钟）

### 1. 创建 Storage Bucket

访问：https://console.cloud.google.com/storage/browser?project=gen-lang-client-0843667030

点击 **"创建存储桶"** 或 **"CREATE BUCKET"**

填写信息：
- **名称**：`gen-lang-client-0843667030-gee-data`
- **位置类型**：Region（区域）
- **位置**：选择离你最近的（如 asia-east1）
- **存储类别**：Standard
- **访问控制**：统一（推荐）

点击 **"创建"**

### 2. 确认服务账号权限

服务账号应该自动有权限，但如果有问题：

1. 进入 bucket 详情页
2. 点击 "权限" 标签
3. 确认 `earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com` 有 **Storage Object Admin** 角色

### 3. 运行导出脚本

```bash
conda activate trajectory-prediction
python scripts/gee_export_to_gcs.py
```

### 4. 等待导出完成

在 Code Editor 的 Tasks 面板查看进度（30分钟-2小时）

### 5. 下载到本地

安装 gsutil（如果还没有）：
```bash
pip install gsutil
```

下载数据：
```bash
gsutil -m cp -r gs://gen-lang-client-0843667030-gee-data/* data/raw/gee/
```

## 预期结果

```
data/raw/gee/
├── bohemian_forest/
│   ├── bohemian_forest_srtm_dem.tif
│   ├── bohemian_forest_worldcover_lulc.tif
│   ├── bohemian_forest_slope.tif
│   └── bohemian_forest_aspect.tif
├── donbas/
│   └── ...
├── carpathians/
│   └── ...
└── scottish_highlands/
    └── ...
```

## 估算成本

- 存储：约 2-5 GB，每月 < $0.10
- 下载：一次性，约 $0.10-0.20
- **总计：几乎免费**
