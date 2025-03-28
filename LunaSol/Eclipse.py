import numpy as np
import spiceypy as spice
from typing import Dict, List
from .utils import *
import os
from .Orbit import Orbit
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Eclipse:
    """天文食(日食/月食)计算基类
    
    提供食预测、误差分析和可视化的通用功能，需要被子类(SolarEclipse/LunarEclipse)实现具体食类型判断逻辑
    
    Attributes:
        orbit (Orbit): 轨道计算实例
        refine_timestep (int): 精细分析时间步长(秒)
        EARTH_RADIUS (float): 地球半径(km)
        MOON_RADIUS (float): 月球半径(km) 
        SUN_RADIUS (float): 太阳半径(km)
        eclipse_types (list): 食类型列表(由子类定义)
    """
    def __init__(self, 
                 reference_file_path=None,
                 refine_timestep=1,
                 bodies=['EARTH','MOON','SUN'],
                 masses ={'EARTH': 5.972e24,'MOON': 7.342e22, 'SUN': 1.989e30}, 
                 kernel_files=['data/de442.bsp', 'data/earth_200101_990827_predict.bpc', 'data/naif0012.tls', 'data/jup346.bsp'],
                 output_dir='output'):
        self.orbit = Orbit(
            bodies=bodies,
            masses=masses,
            kernel_files=kernel_files,
            output_dir=output_dir,     
        )
        self.reference_file = reference_file_path
        self.output_dir = output_dir
        
        self.refine_timestep = refine_timestep
        self.EARTH_RADIUS = EARTH_RADIUS
        self.MOON_RADIUS = MOON_RADIUS
        self.SUN_RADIUS = SUN_RADIUS
        self.eclipse_types = []  # To be defined by subclasses

    def _get_params(self, **kargs):
        """获取参数配置
        
        Args:
            **kargs: 可选参数覆盖默认值
            
        Returns:
            dict: 合并后的参数字典
        """
        params = self.orbit.default_params.copy()
        params.update({k: v for k, v in kargs.items() if v is not None})
        return params

    def _calculate_distance(self, sun, earth, moon):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError

    def _determine_eclipse_type(self, star1, star2, star2_radius, remote_point, remote_angle, near_point, near_angle):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError

    def _analyze_eclipse(self, sun, earth, moon):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError

    def predict_point_eclipse(self, point, sun, earth, moon):
        """
        To be implemented by subclasses
        """
        raise NotImplementedError

    def _refine_analysis(self, start_time, start_state, end_time, time_step=None, use_official_data=False, dynamics_func=None, method=None,
                       rtol=None, atol=None):
        """精细分析食事件时间窗口
        
        Args:
            start_time (str): 开始时间(UTC格式)
            start_state (list): 初始状态向量
            end_time (str): 结束时间(UTC格式) 
            time_step (int, optional): 分析步长(秒)
            use_official_data (bool): 是否使用JPL官方星历
            dynamics_func (function): 动力学函数
            method (str): 积分方法
            rtol (float): 相对误差容限
            atol (float): 绝对误差容限
            
        Returns:
            tuple: (时间列表, 状态列表, 食类型标志列表, 距离列表)
        """
        """
        精细分析
        return:
            times
            states
            flags
            distances
        """
        if method is None:
            use_official_data=True
        time_diff = parse_time(end_time) - parse_time(start_time)
        y_begin = np.array(start_state).astype(float) * 1000
        positions = [self.orbit.bodies.index('EARTH')*6, self.orbit.bodies.index('MOON')*6, self.orbit.bodies.index('SUN')*6]
        times = []
        states = []
        flags = []
        distances = []
        if use_official_data:
            # 使用官方数据
            n_timesteps = int(time_diff.total_seconds()) // time_step + 1

            et_start = spice.str2et(start_time)
            et_times = [et_start + i * time_step for i in range(n_timesteps)]
            for et_time in et_times:

                earth_state, _ = spice.spkezr("EARTH", et_time, self.orbit.frame, "NONE", "SUN")
                moon_state, _ = spice.spkezr("MOON", et_time, self.orbit.frame, "NONE", "SUN")
                sun_state, _ = spice.spkezr("SUN", et_time, self.orbit.frame, "NONE", "SUN")
                
                flag, distance = self._analyze_eclipse(sun_state[:3], earth_state[:3], moon_state[:3])
                if flag != 0:
                    time_str = spice.et2utc(et_time, 'C', 0)
                    times.append(time_str)
                    state = []
                    for body in self.orbit.bodies:
                        body_state, _ =spice.spkezr(body, et_time, self.orbit.frame, "None", "SUN")
                        state.extend(body_state)
                    states.append(state)
                    flags.append(flag)
                    distances.append(distance)
        #使用数值积分方法
        else:
            t_span = (0, time_diff.total_seconds())
            t_eval = np.linspace(0, t_span[1], int(time_diff.total_seconds()) // time_step + 1)
            sol = self.orbit._my_solve_ivp(y_begin, t_span, t_eval, 
                                    method=method, rtol=rtol, atol=atol,
                                    dynamics_func=dynamics_func)
            
            for i in range(len(sol.t)):
                time = add_time(start_time, i * time_step)
                earth_pos = sol.y[positions[0]:positions[0]+3, i]/1000
                moon_pos = sol.y[positions[1]:positions[1]+3, i]/1000
                sun_pos = sol.y[positions[2]:positions[2]+3, i]/1000
                flag, distance = self._analyze_eclipse(sun_pos, earth_pos, moon_pos)
                if flag != 0:
                    times.append(time)
                    states.append(sol.y[:, i]/1000)
                    flags.append(flag)
                    distances.append(distance)
    
        return times, states, flags, distances

    def _refine_generator(self, generator, path_to, time_step=None, use_official_data=False, dynamics_func=None, method=None,
                       rtol=None, atol=None):
        """处理生成器数据并保存到文件
        
        Args:
            generator: 状态生成器
            path_to (str): 输出文件路径
            time_step (int, optional): 分析步长(秒)
            use_official_data (bool): 是否使用JPL官方星历
            dynamics_func (function): 动力学函数
            method (str): 积分方法
            rtol (float): 相对误差容限
            atol (float): 绝对误差容限
            
        Returns:
            str: 输出文件路径
        """
        FILE_HEAD = "Begin time,End time,Eclipse type,Greatest eclipse time"
        for body in self.orbit.bodies:
            FILE_HEAD = FILE_HEAD + f",{body}_Position_X,{body}_Position_Y,{body}_Position_Z,{body}_Velocity_X,{body}_Velocity_Y,{body}_Velocity_Z"
        FILE_HEAD = FILE_HEAD + '\n'
        eclipse_start = None
        eclipse_last_flag = 0
        eclipse_last_state = None
        earth_begin, moon_begin, sun_begin = self.orbit.bodies.index('EARTH')*6, self.orbit.bodies.index('MOON')*6, self.orbit.bodies.index('SUN')*6
        with open(path_to, 'w') as file2:
            file2.write(FILE_HEAD)
            for time, state in generator:
                earth_position = np.array(state[earth_begin:earth_begin+3]).astype(float)
                moon_position = np.array(state[moon_begin:moon_begin+3]).astype(float)
                sun_position = np.array(state[sun_begin:sun_begin+3]).astype(float)
                eclipse_flag, _ = self._analyze_eclipse(sun_position, earth_position, moon=moon_position) 
                if eclipse_flag != eclipse_last_flag and eclipse_flag * eclipse_last_flag == 0:
                    if eclipse_flag!=0:
                        eclipse_start = eclipse_last_state
                    elif eclipse_start:
                        times, states, flags, distances = self._refine_analysis(eclipse_start[0],eclipse_start[1], end_time=time, time_step=time_step, use_official_data=use_official_data, dynamics_func=dynamics_func, method=method, rtol=rtol, atol=atol)
                        distances = np.array(distances)
                        flags = np.array(flags)
                        state_str = get_state_str(states[0])
                        time_pos = np.argmin(distances)
                        flag_pos = np.argmin(flags)
                        file2.write(f'{times[0]},{times[-1]},{self.eclipse_types[flags[time_pos]]},{times[flag_pos]},{state_str}\n')
                        
                        eclipse_start = None
                eclipse_last_flag = eclipse_flag
                eclipse_last_state = (time, state)
        return path_to

    def predict_eclipse(self, startyear, endyear, time_step=None, use_official_data=False, dynamics_funcs=None, methods=None, rtols=None, atols=None):
        """预测指定年份范围内的食事件
        
        Args:
            startyear (int): 起始年份
            endyear (int): 结束年份
            time_step (int, optional): 时间步长(秒)
            use_official_data (bool): 是否使用JPL官方星历数据
            dynamics_funcs (list): 动力学函数列表
            methods (list): 积分方法列表
            rtols (list): 相对误差容限列表
            atols (list): 绝对误差容限列表
            
        Returns:
            list: 生成的预测文件路径列表
        """
        params = self._get_params(
            dynamics_funcs=dynamics_funcs,
            methods=methods,
            rtols=rtols,
            atols=atols
        )
        if time_step is None:
            time_step = self.refine_timestep
        predict_files = []
        
        output_dir = os.path.join(self.output_dir, self.output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        if use_official_data:
            generator = self.orbit.get_official_data(startyear=startyear, endyear=endyear)
            return_path = self._refine_generator(generator, os.path.join(output_dir, 'JPL_data.csv'), time_step=time_step, use_official_data=True)
            predict_files.append(os.path.join(output_dir, 'JPL_data.csv'))
        for dynamics_func in params['dynamics_funcs']:
            for method in params['methods']:
                for rtol in params['rtols']:
                    for atol in params['atols']:
                        generator = self.orbit.predict(startyear, endyear, dynamics_func=dynamics_func, method=method, rtol=rtol, atol=atol)
                        return_path = os.path.join(output_dir, f'{dynamics_func}_{method}_rtol{rtol}_atol{atol}.csv')
                        self._refine_generator(generator, return_path, time_step=time_step, use_official_data=False, dynamics_func=dynamics_func, method=method, rtol=rtol, atol=atol)
                        predict_files.append(return_path)

        return predict_files

    def _get_years_predict(self, filepath: str, startyear: int, endyear: int):
        df = pd.read_csv(filepath)
        # Convert 'Greatest eclipse time' to string first if needed
        df['Year'] = df['Greatest eclipse time'].astype(str).str.extract(r'(\d{4})').astype(int)
        df = df[(df['Year'] >= startyear) & (df['Year'] <= endyear)]
        return df

    def eclipse_loss(self, startyear: int, endyear: int, predict_filepath: str) -> Dict[str, List[float]]:
        """计算预测食数据与标准数据之间的误差
        
        Args:
            startyear: 起始年份
            endyear: 结束年份
            predict_filepath: 预测数据文件路径
            
        Returns:
            Dict: 包含误差统计的字典:
                - 'miss': 未预测到的食数量
                - 'mistake': 类型预测错误的食数量  
                - 'addition': 多预测出的食数量
                - 'loss': 时间误差列表(秒)
        """
        """
        计算预测日食/月食数据与标准数据之间的误差
        
        参数:
            startyear: 起始年份
            endyear: 结束年份
            predict_filepath: 存放预测数据的csv文件
            
        返回:
            包含误差统计的字典:
            {
                'miss': 未预测到的食数量,
                'mistake': 类型预测错误的食数量,
                'addition': 多预测出的食数量,
                'loss': 时间误差列表(秒)
            }
        """
        df_predict = self._get_years_predict(predict_filepath, startyear, endyear)
        df_standard = self._get_years_predict(self.reference_file, startyear, endyear)
        time_predict = df_predict['Greatest eclipse time'].to_numpy() 
        type_predict = df_predict['Eclipse type'].to_numpy()
        time_standard = df_standard['Greatest eclipse time'].to_numpy()
        type_standard = df_standard['Eclipse type'].to_numpy()
        
        miss = 0  # 没有统计到的
        mistake = 0  # 类型错误的
        addition = 0  # 多出来的
        loss = []  # 相差的时间(秒)

        i = 0
        j = 0
        len_predict, len_standard = len(time_predict), len(time_standard)
        
        while i < len_predict and j < len_standard:
            delta = delta_time(time_predict[i], time_standard[j])
            if abs(delta) <= 24 * 3600 * 3:  # 3天内认为是同一事件
                loss.append(abs(delta))
                if type_predict[i] != type_standard[j]:
                    mistake += 1
                i += 1
                j += 1
            elif delta > 0:  # 预测时间晚于标准时间
                miss += 1
                j += 1
            else:  # 预测时间早于标准时间
                addition += 1
                i += 1
        
        # 处理剩余未匹配的事件
        miss += len_standard - j
        addition += len_predict - i
        
        return {
            'miss': miss,
            'mistake': mistake,
            'addition': addition,
            'loss': loss
        }
    
    def generate_video_frame(self, result: dict, time: str, eclipse_type: str, **kargs) -> np.ndarray:
        """生成月食视频帧
        
        参数:
            result: 包含各地月食结果的字典
            time: 当前时间字符串
            type: 月食类型
            **kargs: 其他可选参数，会显示在图像底部
            
        返回:
            np.ndarray: 生成的视频帧图像(RGB格式)
        """
        plt.style.use('seaborn-v0_8-darkgrid')
       
        colors =  ['#FF0000', '#FFFF00', '#0000FF']  # Red, Yellow, Blue for maximum contrast
        point_type = ['s', 's', 's', 's']
        # 创建绘图窗口
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        # 绘制基础地图
        ax.set_global()  # 设置为全球范围
        ax.coastlines(color='white', linewidth=0.5)  # 白色海岸线
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")  # 添加国家边界
        ax.add_feature(cfeature.LAND, facecolor="lightgray")  # 陆地填充颜色
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")  # 海洋填充颜色
        
        # 添加经纬度网格
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
        gl.top_labels = False  # 不显示顶部标签
        gl.right_labels = False  # 不显示右侧标签
        for lat in np.arange(-90, 91, 2):
            for lon in np.arange(-180, 181, 2):
                flag = result[f'{lat}, {lon}'] 
                if flag != 0:
                    ax.plot(lon, lat, marker=point_type[flag], color=colors[flag-1], markersize=2, transform=ccrs.PlateCarree())
        if kargs:
            kargs_text = '\n'.join([f'{k}: {v}' for k,v in kargs.items()])
            plt.figtext(0.02, 0.02, kargs_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.title(f"{self.__class__.__name__[:5]} {eclipse_type} eclipse at {time}\n{' '.join(self.orbit.bodies)}", fontsize=14)
        # 将图形转换为numpy数组
        fig.canvas.draw()
        # 获取图像数据并确保正确的尺寸
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        plt.close()
        return img

    def visualize_error(self, startyear: int, endyear: int, predict_files):
        """可视化预测误差
        
        生成误差类型统计图、时间误差图和误差分布图
        
        Args:
            startyear: 起始年份
            endyear: 结束年份
            predict_files: 预测数据文件路径列表
        """
        """
        可视化预测误差
        
        参数:
            predict_files: 预测数据文件路径列表
        """
        
        # 准备输出目录
        output_dir = os.path.join(self.output_dir, f'{self.output_subdir}_error_visualize')
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集所有预测文件的误差数据
        all_errors = {}
        for pred_file in predict_files:
            errors = self.eclipse_loss(startyear, endyear, pred_file)
            all_errors[' '.join((os.path.basename(pred_file)).split('.')[0].split('_'))] = errors
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = plt.cm.tab20.colors
        
        # 1. 绘制误差类型统计图
        plt.figure(figsize=(12, 6))
        width = 0.25
        x = np.arange(len(all_errors))
        
        for i, (method, errors) in enumerate(all_errors.items()):
            plt.bar(x[i] - width, errors['miss'], width, label='Miss', color=colors[0])
            plt.bar(x[i], errors['mistake'], width, label='Mistake', color=colors[1])
            plt.bar(x[i] + width, errors['addition'], width, label='Addition', color=colors[2])
        
        plt.title(f"Eclipse Prediction Error Types\n{' '.join(self.orbit.bodies)}")
        plt.ylabel('Count')
        plt.xticks(x, all_errors.keys(), rotation=45, ha='right')
        plt.legend(['Miss', 'Mistake', 'Addition'])
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_types_bar.png'), 
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        # 2. 绘制时间误差统计图
        plt.figure(figsize=(12, 6))
        for i, (method, errors) in enumerate(all_errors.items()):
            avg_loss = np.mean(errors['loss']) if errors['loss'] else 0
            plt.bar(method, avg_loss, color=colors[i % len(colors)])
        
        plt.title(f"Average Time Error\n{' '.join(self.orbit.bodies)}")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Time Error (seconds)')
        plt.yscale('log')  # 设置y轴为对数尺度
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_error_bar.png'), 
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        # 2. 绘制时间误差折线图
        plt.figure(figsize=(12, 6))
        for method, errors in all_errors.items():
            if errors['loss']:
                plt.plot(range(len(errors['loss'])), errors['loss'], 'o-', label=method)
        
        plt.title(f"Time Error Distribution\n{' '.join(self.orbit.bodies)}")
        plt.xlabel('Eclipse Event Index')
        plt.ylabel('Time Error (seconds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'time_error_distribution.png'), 
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        # 3. 绘制误差类型饼图
        fig, axes = plt.subplots(1, len(all_errors), figsize=(5*len(all_errors), 5))
        plt.suptitle(f"Error types statistic\n{' '.join(self.orbit.bodies)}")
        if len(all_errors) == 1:
            axes = [axes]
        
        for i, (method, errors) in enumerate(all_errors.items()):
            total = sum([errors['miss'], errors['mistake'], errors['addition']])
            if total == 0:
                # Show empty pie chart with "No Data" label
                axes[i].pie([1], labels=['No Error'], colors=['lightgreen'],
                          wedgeprops={'edgecolor':'lightgreen', 'linewidth':1})
                axes[i].set_title(method)
                continue
                
            sizes = [errors['miss'], errors['mistake'], errors['addition']]
            labels = [f'Miss\n{sizes[0]}', f'Mistake\n{sizes[1]}', f'Addition\n{sizes[2]}']
            
            axes[i].pie(sizes, labels=labels, autopct='%1.1f%%',
                       colors=[colors[0], colors[1], colors[2]])
            axes[i].set_title(method)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_types_pie.png'),
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        print(f'Visualization saved to {output_dir}')


    def visualize_orbit(self, year, filepath, degree=1, time_step=120, **kargs):
        """可视化食期间天体轨道
        
        Args:
            year (int): 年份
            filepath (str): 预测数据文件路径
            degree (int): 经纬度间隔(度)
            time_step (int): 时间步长(秒)
            **kargs: 其他参数传递给_refine_analysis
            
        Generates:
            轨道可视化视频文件
        """
        """可视化食期间天体轨道
        
        Args:
            year: 年份
            filepath: 预测数据文件路径
            degree: 经纬度间隔(度)
            time_step: 时间步长(秒)
            **kargs: 其他参数传递给_refine_analysis
        """
        output_dir = os.path.join(self.output_dir, f'{self.output_subdir}_video')
        os.makedirs(output_dir, exist_ok=True)
        df = self._get_years_predict(filepath=filepath, startyear=year, endyear=year)
        body_columns = []
        for body in self.orbit.bodies:
            body_columns.extend([f'{body}_Position_X',f'{body}_Position_Y',f'{body}_Position_Z',f'{body}_Velocity_X',f'{body}_Velocity_Y',f'{body}_Velocity_Z'])
        for _, row in df.iterrows():
            start_time, end_time , eclipse_type= row['Begin time'], row['End time'], row['Eclipse type']
            start_state = np.array(row[body_columns]).astype(float)
            earth_begin, moon_begin, sun_begin = self.orbit.bodies.index('EARTH')*6, self.orbit.bodies.index('MOON')*6, self.orbit.bodies.index('SUN')*6 
            times, states, _, _ = self._refine_analysis(start_time=start_time,start_state=start_state, end_time=end_time, time_step=time_step, **kargs)
            frames = []
            latitude_range, longitude_range = np.arange(-90, 91, degree), np.arange(-180, 181, degree)
            coordinate_matrix = latlonalt2cartesian(latitude_range, longitude_range)
            for time, state in tqdm(zip(times, states), total=len(times)):
                rotation_matrix = self.orbit.rotation_matrix('ITRF93', self.orbit.frame, time)
                coordinates = np.einsum("ij,jkm->ikm", np.array(rotation_matrix), np.array(coordinate_matrix)) + np.array(state[earth_begin:earth_begin+3])[:, np.newaxis, np.newaxis]
                result ={} 
                for i in range(len(latitude_range)):
                    for j in range(len(longitude_range)):
                        result[f'{-90 + i*degree}, {-180 + j*degree}'] = self.predict_point_eclipse(coordinates[:, i, j], np.array(state[sun_begin:sun_begin+3]), np.array(state[earth_begin:earth_begin+3]), np.array(state[moon_begin:moon_begin+3]))
                frame = self.generate_video_frame(result, time, eclipse_type=eclipse_type, **kargs)
                frames.append(frame)
            if frames:
                generate_video(frames, os.path.join(output_dir, f'{time[0:11]}.mp4'))
    


    def pipeline(self, startyear, endyear, dynamics_funcs=None, methods=None, rtols=None, atols=None):
        """完整食分析流程
        
        1. 预测食事件
        2. 分析预测误差
        3. 可视化结果
        
        Args:
            startyear (int): 起始年份
            endyear (int): 结束年份
            dynamics_funcs (list): 动力学函数列表
            methods (list): 积分方法列表
            rtols (list): 相对误差容限列表
            atols (list): 绝对误差容限列表
        """
        """完整食分析流程
        
        1. 预测食事件
        2. 分析预测误差
        3. 可视化结果
        
        Args:
            startyear: 起始年份
            endyear: 结束年份
            dynamics_funcs: 动力学函数列表
            methods: 积分方法列表
            rtols: 相对误差容限列表
            atols: 绝对误差容限列表
        """
        file_paths = self.predict_eclipse(startyear, endyear, use_official_data=True, dynamics_funcs=dynamics_funcs, methods=methods, rtols=rtols, atols=atols)
        self.visualize_error(startyear=startyear, endyear=endyear, predict_files=file_paths)
        self.visualize_orbit(startyear, file_paths[0], use_official_data=True)
        return file_paths
