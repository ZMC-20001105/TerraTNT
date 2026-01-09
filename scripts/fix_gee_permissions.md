# 修复 Google Earth Engine 权限问题

## 问题描述
服务账号 `earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com` 缺少 Earth Engine 使用权限。

## 解决步骤

### 方法1：通过 Google Cloud Console (推荐)

1. 访问项目 IAM 页面：
   https://console.cloud.google.com/iam-admin/iam?project=gen-lang-client-0843667030

2. 找到服务账号：`earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com`

3. 点击"编辑"按钮，添加以下角色：
   - `Earth Engine Resource Viewer`
   - `Earth Engine Resource Writer`
   - `Service Usage Consumer`

### 方法2：通过 gcloud 命令行

```bash
# 添加 Earth Engine 权限
gcloud projects add-iam-policy-binding gen-lang-client-0843667030 \
    --member="serviceAccount:earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com" \
    --role="roles/earthengine.viewer"

gcloud projects add-iam-policy-binding gen-lang-client-0843667030 \
    --member="serviceAccount:earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com" \
    --role="roles/earthengine.writer"

gcloud projects add-iam-policy-binding gen-lang-client-0843667030 \
    --member="serviceAccount:earth-engine-service@gen-lang-client-0843667030.iam.gserviceaccount.com" \
    --role="roles/serviceusage.serviceUsageConsumer"
```

### 方法3：启用 Earth Engine API

确保项目已启用 Earth Engine API：
https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project=gen-lang-client-0843667030

## 验证

权限配置完成后，运行以下命令验证：

```bash
python scripts/test_gee_connection.py
```

## 注意事项

- 权限更改可能需要几分钟才能生效
- 确保项目已启用 Earth Engine API
- 如果仍有问题，可能需要申请 Earth Engine 访问权限
