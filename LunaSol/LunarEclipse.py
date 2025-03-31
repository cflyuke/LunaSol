from .Eclipse import Eclipse
import math
import numpy as np
from .utils import *

class LunarEclipse(Eclipse):
    """月食计算类
    
    继承自Eclipse基类，实现月食特定计算逻辑
    
    Attributes:
        eclipse_types (list): 月食类型列表
            - 'Total': 月全食
            - 'Partial': 月偏食
            - 'Penumbral': 半影月食
        output_subdir (str): 输出子目录名
    """
    def __init__(self, **kwargs):
        kwargs['reference_file_path'] = kwargs.get('reference_file_path', 'data/lunar_eclipse_data.csv')
        super().__init__(**kwargs)
        self.eclipse_types = ['No', 'Total', 'Partial', 'Penumbral']
        self.output_subdir = 'Lunar_eclipse'


    def _calculate_distance(self, sun, earth, moon):
        """计算月食相关距离
        
        Args:
            sun (array): 太阳位置向量(km)
            earth (array): 地球位置向量(km)
            moon (array): 月球位置向量(km)
            
        Returns:
            float: 月球到地球阴影锥的最小距离(km)
        """
        """
        计算月球到地球阴影轴的距离
        From [https://eclipse.gsfc.nasa.gov/LEhelp/LEglossary.html]
        For lunar eclipses, Greatest Eclipse (GE) is defined as the instant when the Moon passes closest to the axis of Earth's shadow.
        """
        return distance(earth, moon) * math.sin(angle(moon, earth, sun))

    def _determine_eclipse_type(self, star1, star2, star2_radius, remote_point, remote_angle, near_point, near_angle):
        """确定月食类型 
        
        Args:
            star1 (array): 地球位置向量(km)
            star2 (array): 月球位置向量(km)
            star2_radius (float): 地球半径(km)
            remote_point (array): 本影锥锥顶位置向量(km)
            remote_angle (float): 本影锥锥角(rad)
            near_point (array): 半影锥锥顶位置向量(km)
            near_angle (float): 半影锥锥角(rad)
            
        Returns:
            int: 月食类型索引(0:无食,1:全食,2:偏食,3:半影食)
        """
        star2_near_angle = angle(star2, near_point, star1)
        delta_near = np.arcsin(star2_radius / distance(near_point, star2))
        if distance(star1, near_point)*np.cos(near_angle) > distance(star2, near_point):
            return 0
        if star2_near_angle - delta_near <= near_angle: 
            star2_remote_angle = angle(star2, remote_point, star1)
            delta_remote = np.arcsin(star2_radius / distance(remote_point, star2))
            if star2_remote_angle + delta_remote <= remote_angle:
                return 1
            elif star2_remote_angle - delta_remote <= remote_angle:
                return 2
            else:
                return 3
        else:
            return 0

    def _analyze_eclipse(self, sun, earth, moon):
        """分析月食事件
        
        Args:
            sun (array): 太阳位置向量(km)
            earth (array): 地球位置向量(km)
            moon (array): 月球位置向量(km)
            
        Returns:
            tuple: (食类型标志, 最小距离)
        """
        remote_angle, remote_point = calculate_remote_point(sun, earth, self.SUN_RADIUS, self.EARTH_RADIUS)
        near_angle, near_point = calculate_near_point(sun, earth, self.SUN_RADIUS, self.EARTH_RADIUS)
        flag = self._determine_eclipse_type(star1=earth, star2=moon, star2_radius=self.MOON_RADIUS, 
                                         remote_point=remote_point, remote_angle=remote_angle, 
                                         near_point=near_point, near_angle=near_angle)
        distance = self._calculate_distance(sun=sun, earth=earth, moon=moon) if flag != 0 else float('inf')
        return flag, distance
    
    def predict_point_eclipse(self, point, sun, earth, moon):
        """预测某一点的月食情况
        
        Args:
            point: 月球表面点位置(km)
            sun: 太阳位置(km)
            earth: 地球位置(km)
            moon: 月球位置(km)
            
        Returns:
            int: 月食类型索引
        """
        earth_remote_angle, _ = calculate_remote_point(sun, earth, self.SUN_RADIUS, self.EARTH_RADIUS)
        theta = math.pi - angle(point, earth, sun)
        if theta + earth_remote_angle > math.pi /2:
            return 0
        earth_moon_remote_angle, _ =calculate_remote_point(earth, moon, self.EARTH_RADIUS, self.MOON_RADIUS)
        if angle(point, earth, moon) + earth_moon_remote_angle > math.pi/2:
            return 0
        
        return 1

if __name__=='__main__':
    eclipse = LunarEclipse(
        bodies=['EARTH', 'MOON', 'SUN', 'JUPITER'], 
        masses={ 'EARTH': 5.972e+24,'MOON': 7.342e+22,'SUN': 1.989e+30, 'JUPITER':1.898e27}, 
        kernel_files=['data/de442.bsp', 'data/earth_200101_990827_predict.bpc', 'data/naif0012.tls', 'data/jup346.bsp'], 
        output_dir='output'
    )
    eclipse.visualize_orbit(2032, 'output/Lunar_eclipse/standard_RK45_rtol1e-06_atol1e-06.csv', degree=2, time_step=120, use_official_data=False, method='RK45', dynamics_func='standard', rtol=1e-6, atol=1e-6)
