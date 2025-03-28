from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
EARTH_RADIUS = 6371
MOON_RADIUS = 1737
SUN_RADIUS = 696340

def parse_time(time_str: str, time_format: str = '%Y %b %d %H:%M:%S') -> datetime:
    """解析时间字符串为datetime对象(不区分大小写)
    
    参数:
        time_str: 时间字符串
        time_format: 时间格式字符串(默认: '%Y %b %d %H:%M:%S')
        
    返回:
        datetime: 解析后的datetime对象
    """
    return datetime.strptime(time_str.upper(), time_format)

def add_time(time_str: str, delta_seconds: float, time_format: str = "%Y %b %d %H:%M:%S") -> str:
    """在给定时间字符串上增加指定秒数
    
    参数:
        time_str: 时间字符串(不区分大小写)
        delta_seconds: 要增加的秒数
        time_format: 时间格式字符串(默认: "%Y %b %d %H:%M:%S")
        
    返回:
        str: 增加时间后的新时间字符串
    """
    time_obj = datetime.strptime(time_str.upper(), time_format)
    new_time_obj = time_obj + timedelta(seconds=delta_seconds)
    return new_time_obj.strftime(time_format)

def get_state_str(state: list) -> str:
    """将状态列表转换为逗号分隔的字符串
    
    参数:
        state: 包含各种状态值的列表
        
    返回:
        str: 逗号分隔的状态字符串
    """
    return ','.join([str(x) for x in state])

def distance(point1: list, point2: list) -> float:
    """计算两点之间的欧几里得距离
    
    参数:
        point1: 第一个点的坐标[x1,y1,z1]
        point2: 第二个点的坐标[x2,y2,z2]
        
    返回:
        float: 两点之间的距离
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def angle(point1: list, point2: list, point3: list) -> float:
    """计算三点形成的角度(点2为顶点)
    
    参数:
        point1: 第一个点坐标
        point2: 顶点坐标
        point3: 第三个点坐标
        
    返回:
        float: 三点形成的角度(弧度)
    """
    point1, point2, point3 = np.array(point1), np.array(point2), np.array(point3)
    vector1 = point1 - point2
    vector2 = point3 - point2
    return np.arccos(np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))

def calculate_remote_point(star1: np.ndarray, star2: np.ndarray, star1_radius: float, star2_radius: float) -> tuple:
    """计算两个天体之间的远点(remote point)和视角
    
    参数:
        star1: 第一个天体的坐标(3D数组)
        star2: 第二个天体的坐标(3D数组) 
        star1_radius: 第一个天体的半径(km)
        star2_radius: 第二个天体的半径(km)
        
    返回:
        tuple: (remote_angle, remote_point)
            remote_angle: 远点视角(弧度)
            remote_point: 远点坐标(3D数组)
    """
    star2 = np.array(star2)
    star1 = np.array(star1)
    star2_vector = star2 - star1
    remote_point_vector = star2_vector * star1_radius / (star1_radius - star2_radius)
    remote_point = star1 + remote_point_vector
    remote_angle = np.arcsin((star1_radius - star2_radius) / distance(star1, star2))
    return remote_angle, remote_point

def calculate_near_point(star1: np.ndarray, star2: np.ndarray, star1_radius: float, star2_radius: float) -> tuple:
    """计算两个天体之间的近点(near point)和视角
    
    参数:
        star1: 第一个天体的坐标(3D数组)
        star2: 第二个天体的坐标(3D数组)
        star1_radius: 第一个天体的半径(km)
        star2_radius: 第二个天体的半径(km)
        
    返回:
        tuple: (near_angle, near_point)
            near_angle: 近点视角(弧度)
            near_point: 近点坐标(3D数组)
    """
    star2 = np.array(star2)
    star1 = np.array(star1)
    star2_vector = star2 - star1
    near_point_vector = star2_vector * star1_radius / (star1_radius + star2_radius)
    near_point = star1 + near_point_vector
    near_angle = np.arcsin((star1_radius + star2_radius) / distance(star1, star2))
    return near_angle, near_point

def delta_time(time_str1: str, time_str2: str, time_format: str = "%Y %b %d %H:%M:%S") -> float:
    """计算两个时间字符串之间的时间差(秒)
    
    参数:
        time_str1: 第一个时间字符串
        time_str2: 第二个时间字符串
        time_format: 时间格式字符串(默认: "%Y %b %d %H:%M:%S")
        
    返回:
        float: 时间差(秒)，正数表示time_str1晚于time_str2
    """
    time1 = datetime.strptime(time_str1.upper(), time_format)
    time2 = datetime.strptime(time_str2.upper(), time_format)
    return (time1 - time2).total_seconds()

def latlonalt2cartesian(latitude: np.ndarray, longitude: np.ndarray, altitude: float = 0, flattening: float = 1/298.257) -> np.ndarray:
    """将经纬度和海拔转换为ITRF93坐标系下的笛卡尔坐标
    
    参数:
        latitude: 纬度数组(度)
        longitude: 经度数组(度)
        altitude: 海拔高度(km)，默认为0
        flattening: 地球扁平率，默认为1/298.257
        
    返回:
        np.ndarray: 3D数组，形状为(3, len(latitude), len(longitude))，
                    包含转换后的笛卡尔坐标[x,y,z]
    """
  
    # Reshape for broadcasting
    lat = latitude.reshape(-1, 1)
    lon = longitude.reshape(1, -1)
    ones_rows = np.ones_like(lon)

    e2 = flattening*(2 - flattening)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = EARTH_RADIUS / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N*(1 - e2) + altitude) * np.sin(lat_rad) * ones_rows
    
    return np.stack([x, y, z], axis=0)

def generate_video(frames: list, video_path: str) -> None:
    """将多帧图像合成为MP4视频
    
    参数:
        frames: 视频帧列表(每个元素为np.ndarray)
        video_path: 输出视频文件路径
        
    返回:
        None
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
