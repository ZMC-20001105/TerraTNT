"""
全局常量定义
集中管理所有硬编码的常量值
"""
from enum import Enum
from typing import Dict, List

# ============================================================
# 地理坐标系统常量
# ============================================================

class CoordinateSystem(Enum):
    """坐标系统枚举"""
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    UTM_30N = "EPSG:32630"
    UTM_33N = "EPSG:32633"
    UTM_35N = "EPSG:32635"
    UTM_37N = "EPSG:32637"


# ============================================================
# LULC 分类常量
# ============================================================

class LULCClass(Enum):
    """ESA WorldCover 土地利用分类"""
    TREE_COVER = 10
    SHRUBLAND = 20
    GRASSLAND = 30
    CROPLAND = 40
    BUILT_UP = 50
    BARE_SPARSE = 60
    PERMANENT_WATER = 80
    HERBACEOUS_WETLAND = 90
    MOSS_LICHEN = 100


LULC_NAMES: Dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    100: "Moss and lichen",
}

LULC_SHORT_NAMES: Dict[int, str] = {
    10: "Forest",
    20: "Shrub",
    30: "Grass",
    40: "Crop",
    50: "Urban",
    60: "Bare",
    80: "Water",
    90: "Wetland",
    100: "Moss",
}

# 可通行性（0=禁止，1=完全通行）
LULC_TRAVERSABILITY: Dict[int, float] = {
    10: 0.7,   # 森林：较慢
    20: 0.8,   # 灌木：中等
    30: 1.0,   # 草地：基准
    40: 0.9,   # 农田：较快
    50: 0.0,   # 建筑：禁止
    60: 0.6,   # 裸地：慢
    80: 0.0,   # 水体：禁止
    90: 0.3,   # 湿地：很慢
    100: 0.5,  # 苔藓：慢
}


# ============================================================
# 物理常量
# ============================================================

EARTH_RADIUS = 6371000.0  # 地球半径（米）
GRAVITY = 9.81  # 重力加速度（m/s²）

# 速度限制
MAX_VEHICLE_SPEED = 30.0  # 最大车辆速度（m/s）
MIN_VEHICLE_SPEED = 0.1   # 最小车辆速度（m/s）

# 坡度限制
MAX_TRAVERSABLE_SLOPE = 45.0  # 最大可通行坡度（度）
STEEP_SLOPE_THRESHOLD = 30.0  # 陡坡阈值（度）


# ============================================================
# 数据处理常量
# ============================================================

# 时间相关
GPS_SAMPLING_RATE = 4  # Hz
IMU_SAMPLING_RATE = 100  # Hz
TRAJECTORY_SAMPLING_RATE = 4  # Hz（统一采样率）

# 轨迹处理
MIN_TRAJECTORY_LENGTH = 100  # 最小轨迹点数
TRAJECTORY_SMOOTH_WINDOW = 5  # 平滑窗口大小

# 环境特征
NUM_ENVIRONMENT_CHANNELS = 18  # 环境特征通道数
ENVIRONMENT_WINDOW_SIZE = 256  # 局部环境窗口大小（像素）

# TerraTNT 数据
HISTORY_LENGTH = 240  # 历史轨迹长度（60s * 4Hz）
PREDICTION_LENGTH = 240  # 预测轨迹长度（60min / 15s）
PREDICTION_INTERVAL = 15  # 预测间隔（秒）


# ============================================================
# 模型常量
# ============================================================

# 随机种子
RANDOM_SEED = 42

# 批次大小
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

# 训练参数
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# 早停
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001


# ============================================================
# 评估指标常量
# ============================================================

# Miss Rate 阈值（米）
MR_THRESHOLDS: List[float] = [100.0, 200.0, 500.0]

# 评估时间点（秒）
EVAL_TIME_HORIZONS: List[int] = [15, 30, 60, 120, 300, 600, 1800, 3600]


# ============================================================
# 文件格式常量
# ============================================================

class FileFormat(Enum):
    """文件格式枚举"""
    GEOTIFF = ".tif"
    NUMPY = ".npy"
    PICKLE = ".pkl"
    HDF5 = ".h5"
    CSV = ".csv"
    JSON = ".json"
    YAML = ".yaml"


# ============================================================
# 日志级别常量
# ============================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================
# 颜色常量（用于终端输出）
# ============================================================

class TerminalColor:
    """终端颜色代码"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ============================================================
# 单位转换常量
# ============================================================

# 角度转换
DEG_TO_RAD = 0.017453292519943295  # π/180
RAD_TO_DEG = 57.29577951308232     # 180/π

# 速度转换
MS_TO_KMH = 3.6   # m/s to km/h
KMH_TO_MS = 0.277777778  # km/h to m/s

# 距离转换
M_TO_KM = 0.001
KM_TO_M = 1000.0


# ============================================================
# 数据验证常量
# ============================================================

# GPS 数据有效性
GPS_MIN_LATITUDE = -90.0
GPS_MAX_LATITUDE = 90.0
GPS_MIN_LONGITUDE = -180.0
GPS_MAX_LONGITUDE = 180.0
GPS_MAX_HORIZONTAL_ACCURACY = 10.0  # 米

# IMU 数据有效性
IMU_MAX_ACCELERATION = 50.0  # m/s²
IMU_MAX_GYRO = 10.0  # rad/s


# ============================================================
# 缓存配置常量
# ============================================================

CACHE_MAX_SIZE = 1073741824  # 1GB
CACHE_TTL = 3600  # 缓存过期时间（秒）


# ============================================================
# 网络请求常量
# ============================================================

REQUEST_TIMEOUT = 300  # 请求超时（秒）
MAX_RETRIES = 5  # 最大重试次数
RETRY_BACKOFF_FACTOR = 2  # 重试退避因子
