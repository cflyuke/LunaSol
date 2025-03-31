from .Eclipse import Eclipse
import math
import numpy as np
from .utils import *


class SolarEclipse(Eclipse):
    """日食计算类
    
    继承自Eclipse基类，实现日食特定计算逻辑
    
    Attributes:
        eclipse_types (list): 日食类型列表
            - 'Total': 日全食
            - 'Annular': 日环食
            - 'Partial': 日偏食
            - 'Hybrid': 全环食
        output_subdir (str): 输出子目录名
    """
    def __init__(self, **kwargs):
        kwargs['reference_file_path'] = kwargs.get('reference_file_path', 'data/solar_eclipse_data.csv')
        super().__init__(**kwargs)
        self.eclipse_types = ['No', 'Total', 'Partial', 'Annular', 'Hybrid']
        self.output_subdir = 'Solar_eclipse'

    def _calculate_distance(self, sun, earth, moon):
        """计算日食相关距离
        
        Args:
            sun (array): 太阳位置向量(km)
            earth (array): 地球位置向量(km)
            moon (array): 月球位置向量(km)
            
        Returns:
            tuple: (太阳-月球距离, 地球-月球距离, 太阳-地球距离)(km)
        """
        """
        From [https://eclipse.gsfc.nasa.gov/SEhelp/SEglossary.html]
        For solar eclipses, Greatest Eclipse (GE) is defined as the instant when the axis of the Moon's shadow cone passes closest to Earth's center.
        """
        return distance(moon, earth) * math.sin(angle(earth, moon, sun))

    def _determine_eclipse_type(self, star1, star2, star2_radius, remote_point, remote_angle, near_point, near_angle):
        """确定日食类型
        
        Args:
            star1 (array): 月球位置向量(km)
            star2 (array): 地球位置向量(km)
            star2_radius (float): 地球半径(km)
            remote_point (array): 本影锥锥顶位置向量(km)
            remote_angle (float): 本影锥锥角(rad)
            near_point (array): 半影锥锥顶位置向量(km)
            near_angle (float): 半影锥锥角(rad)
            
        Returns:
            tuple: 食类型标志
        """
        if angle(star2, star1, near_point) < math.pi/2:
            return 0
        if star2_radius >= distance(remote_point, star2):
            if angle(star2, remote_point, star1) <= math.pi/2:
                return 1
            else:
                return 4
        
        star2_remote_angle = angle(star2, remote_point, star1)
        delta_remote = np.arcsin(star2_radius / distance(remote_point, star2))
        star2_near_angle = angle(star2, near_point, star1)
        delta_near = np.arcsin(star2_radius / distance(near_point, star2))
        if star2_remote_angle - delta_remote < remote_angle:
            return 1
        elif star2_near_angle - delta_near  < near_angle and star2_remote_angle + remote_angle + delta_remote >= math.pi:
            if star2_remote_angle < math.pi / 2:
                return 2
            else:
                return 3
        elif star2_near_angle - delta_near  < near_angle:
            return 2
        elif star2_remote_angle + remote_angle + delta_remote >= math.pi:
            return 3
        else:
            return 0

    def _analyze_eclipse(self, sun, earth, moon):
        """分析日食事件
        
        Args:
            sun (array): 太阳位置向量(km)
            earth (array): 地球位置向量(km)
            moon (array): 月球位置向量(km)
            
        Returns:
            tuple: (食类型标志, 最小距离)
        """
        remote_angle, remote_point = calculate_remote_point(sun, moon, self.SUN_RADIUS, self.MOON_RADIUS)
        near_angle, near_point = calculate_near_point(sun, moon, self.SUN_RADIUS, self.MOON_RADIUS)
        flag = self._determine_eclipse_type(moon, earth, star2_radius=self.EARTH_RADIUS, remote_point=remote_point, remote_angle=remote_angle, near_point=near_point, near_angle=near_angle)
        distance = self._calculate_distance(sun=sun, earth=earth, moon=moon) if flag != 0 else float('inf')
        return flag, distance
    
    def predict_point_eclipse(self, point, sun, earth, moon):
        """预测某一点的日食情况
        
        Args:
            point: 地面点位置(km)
            sun: 太阳位置(km)
            earth: 地球位置(km)
            moon: 月球位置(km)
            
        Returns:
            dict: 包含日食信息的字典
        """
        moon_near_angle, moon_near_point = calculate_near_point(star1=sun, star2=moon, star1_radius=self.SUN_RADIUS, star2_radius=self.MOON_RADIUS)
        moon_remote_angle, moon_remote_point = calculate_remote_point(star1=sun, star2=moon, star1_radius=self.SUN_RADIUS, star2_radius=self.MOON_RADIUS)

        earth_remote_angle, _ = calculate_remote_point(star1=sun, star2=earth, star1_radius=self.SUN_RADIUS, star2_radius=self.EARTH_RADIUS)

        point_near_angle = angle(point, moon_near_point, moon)
        point_remote_angle = angle(point, moon_remote_point, moon)

        #判断是否进入半影区域
        if point_near_angle >= moon_near_angle:
            return 0
    
        #判断该点是否会被光照到, 如果本来就不会被光照到，就不用考虑日食了
        theta = math.pi - angle(point, earth, sun)
        if theta + earth_remote_angle < math.pi/2:
            return 0
        #判断是在哪个区域进而判断是什么种类的日食
        if point_remote_angle <= moon_remote_angle:
            return 1
        elif point_remote_angle >= math.pi - moon_remote_angle:
            return 3
        else:
            return 2
    


if __name__ == '__main__':
    eclipse = SolarEclipse()
    eclipse.predict(2025, 'output/Solar_eclipse_predict/JPL_data.csv', use_official_data=True)
