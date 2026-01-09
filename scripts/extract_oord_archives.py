"""
批量解压 OORD GPS/IMU 压缩包。

注意：官方文件名是 *.zip，但实际是 tar.gz（gzip 压缩的 tar 包）。

解压结构（方案 A）：
  data/oord_extracted/<region>/gps_XX/MicrostrainMIP/{gps.csv, imu.csv}

同时清理 gdown 遗留的 *.part 文件。
"""

import sys
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    # 防止 tar 里带有绝对路径或 ..
    name = name.lstrip("/\\")
    name = name.replace("..", "")
    return name


def _safe_extract_tar(tar: tarfile.TarFile, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)

    for member in tar.getmembers():
        member.name = _safe_name(member.name)
        target_path = extract_dir / member.name
        if not str(target_path.resolve()).startswith(str(extract_dir.resolve())):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")

    tar.extractall(path=extract_dir)


def find_archives(oord_root: Path) -> List[Tuple[str, Path]]:
    """返回 (region, archive_path) 列表"""
    results: List[Tuple[str, Path]] = []
    for region_dir in sorted(oord_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for fp in sorted(region_dir.glob("gps_*.zip")):
            results.append((region_dir.name, fp))
    return results


def cleanup_part_files(oord_root: Path) -> List[Path]:
    removed: List[Path] = []
    for fp in oord_root.rglob("*.part"):
        try:
            fp.unlink()
            removed.append(fp)
        except Exception as e:
            logger.warning(f"删除失败: {fp} ({e})")
    return removed


def extract_one_archive(region: str, archive_path: Path, out_root: Path) -> Path:
    """解压单个 tar.gz（伪装成 .zip）的包。返回实际解压目录。"""
    tag = archive_path.stem  # gps_01
    extract_dir = out_root / region / tag
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 直接按 tar.gz 处理
    with tarfile.open(archive_path, mode="r:gz") as tar:
        _safe_extract_tar(tar, extract_dir)

    return extract_dir


def normalize_layout(extract_dir: Path) -> Path:
    """将解压结果统一为 <extract_dir>/MicrostrainMIP/... 如果 tar 里多套一层则拉平。"""
    # 理想结构：extract_dir/MicrostrainMIP/gps.csv
    mip = extract_dir / "MicrostrainMIP"
    if mip.exists() and mip.is_dir():
        return extract_dir

    # 若 tar 中只有 MicrostrainMIP 目录但被包在一层目录里
    candidates = list(extract_dir.glob("*/MicrostrainMIP"))
    if candidates:
        src_mip = candidates[0]
        # 将该层目录内容移动到 extract_dir
        parent = src_mip.parent
        for item in parent.iterdir():
            dst = extract_dir / item.name
            if dst.exists():
                continue
            shutil.move(str(item), str(dst))
        # 清理空目录
        try:
            parent.rmdir()
        except Exception:
            pass
        return extract_dir

    return extract_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    oord_root = Path(cfg.get("paths.raw_data.oord"))
    out_root = Path(cfg.get("paths.raw_data.oord_extracted"))
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"OORD 原始目录: {oord_root}")
    logger.info(f"OORD 解压目录: {out_root}")

    # 清理 part
    removed = cleanup_part_files(oord_root)
    if removed:
        logger.info(f"清理 .part 文件: {len(removed)}")

    archives = find_archives(oord_root)
    if not archives:
        logger.error("未找到任何 gps_*.zip")
        raise SystemExit(1)

    ok = 0
    for region, archive_path in archives:
        try:
            logger.info(f"解压 {region}/{archive_path.name} ...")
            extract_dir = extract_one_archive(region, archive_path, out_root)
            extract_dir = normalize_layout(extract_dir)

            gps_csv = extract_dir / "MicrostrainMIP" / "gps.csv"
            imu_csv = extract_dir / "MicrostrainMIP" / "imu.csv"

            if not gps_csv.exists() or not imu_csv.exists():
                logger.warning(f"结构异常: {extract_dir} (gps.csv exists={gps_csv.exists()}, imu.csv exists={imu_csv.exists()})")
            else:
                ok += 1

        except Exception as e:
            logger.error(f"解压失败: {region}/{archive_path.name} ({e})")

    logger.info("=" * 60)
    logger.info(f"解压完成: {ok}/{len(archives)} 包包含 MicrostrainMIP/gps.csv 和 imu.csv")


if __name__ == "__main__":
    main()
