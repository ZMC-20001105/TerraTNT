#!/usr/bin/env python3
"""
下载 OSM PBF 文件的辅助脚本
使用 requests 库，支持断点续传和进度显示
"""
import argparse
import sys
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    下载文件，支持断点续传
    
    Args:
        url: 下载链接
        output_path: 输出文件路径
        chunk_size: 每次下载的块大小
    """
    # 检查是否已存在部分下载的文件
    resume_byte_pos = 0
    if output_path.exists():
        resume_byte_pos = output_path.stat().st_size
        print(f"发现已下载 {resume_byte_pos / 1024 / 1024:.1f} MB，尝试断点续传...")
    
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'identity',
        'Connection': 'keep-alive',
    }
    
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    # 发起请求
    print(f"正在连接 {url}...")
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    
    # 检查响应
    if response.status_code == 416:
        print("文件已完整下载")
        return True
    
    if response.status_code not in (200, 206):
        print(f"下载失败: HTTP {response.status_code}")
        print(f"响应内容: {response.text[:500]}")
        return False
    
    # 获取文件总大小
    total_size = int(response.headers.get('content-length', 0))
    if total_size == 0:
        print("警告: 服务器未返回文件大小")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"响应头: {dict(response.headers)}")
        return False
    
    if response.status_code == 206:
        total_size += resume_byte_pos
    
    print(f"文件总大小: {total_size / 1024 / 1024:.1f} MB")
    
    # 下载文件
    mode = 'ab' if resume_byte_pos > 0 else 'wb'
    with open(output_path, mode) as f:
        with tqdm(
            total=total_size,
            initial=resume_byte_pos,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # 验证下载
    final_size = output_path.stat().st_size
    print(f"\n下载完成: {final_size / 1024 / 1024:.1f} MB")
    
    if total_size > 0 and final_size != total_size:
        print(f"警告: 文件大小不匹配 (期望 {total_size}, 实际 {final_size})")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='下载 OSM PBF 文件')
    parser.add_argument('--url', type=str, required=True, help='下载链接')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--chunk-size', type=int, default=8192, help='下载块大小')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = download_file(args.url, output_path, args.chunk_size)
    
    if success:
        print("✓ 下载成功")
        sys.exit(0)
    else:
        print("✗ 下载失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
