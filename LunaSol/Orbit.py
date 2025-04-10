import spiceypy as spice
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .utils import parse_time, add_time
from tqdm import tqdm
import csv


# Sun: 1.988470e+30 kg  
# Earth: 5.972168e+24 kg  
# Moon: 7.346303e+22 kg  
# Jupiter: 1.898187e+27 kg  
# Venus: 4.867468e+24 kg


class Orbit:
    def __init__(self, 
                 bodies=['EARTH', 'MOON', 'SUN'], 
                 observer='SUN',
                 frame='ECLIPJ2000',
                 masses={'EARTH': 5.972168e24,'MOON': 7.346303e22, 'SUN': 1.988470e30},
                 kernel_files=['data/de442.bsp', 'data/earth_200101_990827_predict.bpc', 'data/naif0012.tls', 'data/jup346.bsp'],
                 output_dir='output',
                 ):
        """
        初始化轨道计算类
        
        参数:
            bodies: 要计算的天体列表 (SPICE ID格式)
            masses: 要计算的天体质量 (单位为kg)
            kernel_files: SPICE内核文件列表
            output_dir: 输出文件根目录
        """
        self.bodies = bodies
        self.masses = masses
        self.kernel_files = kernel_files
        self.output_dir = output_dir

        self.frame = frame
        self.observer = observer

        self.G = 6.67430e-11

        self.default_params = {
            'dynamics_funcs': ['standard'],
            'methods': ['RK45'],
            'rtols': [1e-6],
            'atols': [1e-6],
            'stepsize': 600
        }
        self.dynamics_funcs = {
            'standard': self._N_body_dynamics,
        }   
        

        self._load_kernels()
    
    def _load_kernels(self):
        """加载SPICE内核文件"""
        for file_path in self.kernel_files:
            spice.furnsh(file_path)
    
    def _unload_kernels(self):
        """卸载SPICE内核"""
        spice.kclear()
    
    def _N_body_dynamics(self, t, y):
        """N-body动力学方程
        计算N个天体在相互引力作用下的运动方程
        
        参数:
            t: 当前时间(秒)
            y: 状态向量(6*N维数组，包含N个天体的位置和速度)
            
        返回:
            6*N维导数向量，包含:
            - 每个天体的速度分量(vx,vy,vz)
            - 每个天体的加速度分量(ax,ay,az)
            
        数学原理:
            基于牛顿万有引力定律:
            F = G*m1*m2/r^2
            加速度 a = F/m = G*M/r^2
        """
        n_bodies = len(self.bodies)
        state_size = 6 * n_bodies  # 6 state variables per body (x,y,z,vx,vy,vz)
        
        # Split state vector into positions and velocities for each body
        positions = np.array([y[0:state_size:6], y[1:state_size:6], y[2:state_size:6]])
        velocities = np.array([y[3:state_size:6], y[4:state_size:6], y[5:state_size:6]])
        
        # Initialize accelerations
        accelerations = np.zeros((3, n_bodies))
        
        # Calculate gravitational forces between all pairs of bodies
        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                # Distance vector between bodies i and j
                r_vec = positions[:,j] - positions[:,i]
                r = np.linalg.norm(r_vec)
                
                # Force magnitudes
                force_mag = self.G * self.masses[self.bodies[j]] / r**3
                accelerations[:,i] += force_mag * r_vec
                
                force_mag = self.G * self.masses[self.bodies[i]] / r**3 
                accelerations[:,j] -= force_mag * r_vec
        
        # Build derivative vector [vx1,vy1,vz1,ax1,ay1,az1, vx2,...]
        derivative = []
        for i in range(n_bodies):
            derivative.extend(velocities[:,i])
            derivative.extend(accelerations[:,i])
            
        return derivative
    
    def _my_solve_ivp(self, y0, t_span, t_eval, method, rtol, atol, dynamics_func):
        """自定义积分器
        对solve_ivp的封装，添加了坐标系转换功能
        
        参数:
            y0: 初始状态向量(6*N维数组，N为天体数量)
            t_span: 积分时间范围(秒)
            t_eval: 需要计算的时间点数组
            method: 积分方法(RK45/DOP853等)
            rtol: 相对误差容限
            atol: 绝对误差容限
            dynamics_func: 动力学函数名称
            
        返回:
            solve_ivp的Solution对象，包含:
            - t: 时间点数组
            - y: 状态向量数组(已转换为观测中心坐标系)
            - sol: 积分器状态(可选)
        """
        func = self.dynamics_funcs.get(dynamics_func, self._N_body_dynamics)
        sol = solve_ivp(func, t_span, y0=y0,
                       t_eval=t_eval, method=method, rtol=rtol, atol=atol)
        # 转换为观测中心坐标系
        observer_idx = self.bodies.index(self.observer)
        observer_pos_start = observer_idx * 6
        observer_vel_start = observer_idx * 6 + 3
        
        for i in range(len(self.bodies)):
            if i == observer_idx:
                continue
            pos_start = i * 6
            vel_start = i * 6 + 3
            sol.y[pos_start:pos_start+3] = sol.y[pos_start:pos_start+3] - sol.y[observer_pos_start:observer_pos_start+3]
            sol.y[vel_start:vel_start+3] = sol.y[vel_start:vel_start+3] - sol.y[observer_vel_start:observer_vel_start+3]
        
        # 设置观测中心状态为0
        sol.y[observer_pos_start:observer_pos_start+3] = 0.0
        sol.y[observer_vel_start:observer_vel_start+3] = 0.0
        return sol
    


    def _get_params(self, **kwargs):
        """获取参数并自动填充默认值
        
        参数:
            **kwargs: 关键字参数，用于覆盖默认参数，包括:
                - methods: 积分方法列表
                - rtols: 相对误差容限列表
                - atols: 绝对误差容限列表
                - dynamics_funcs: 动力学函数列表
                - stepsize: 积分步长(秒)
                
        返回:
            包含所有参数的字典，结构为:
            {
                'methods': 积分方法列表,
                'rtols': 相对误差容限列表,
                'atols': 绝对误差容限列表,
                'dynamics_funcs': 动力学函数列表,
                'stepsize': 积分步长(秒)
            }
            
        功能:
            1. 复制默认参数
            2. 用kwargs中的非None值覆盖默认参数
            3. 返回合并后的参数字典
        """
        params = self.default_params.copy()
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def update_dynamics_funcs(self, dynamics_funcs_dict):
        """更新动力学函数字典
        允许动态添加自定义的动力学函数
        
        参数:
            dynamics_funcs_dict: 要添加的动力学函数字典，格式为:
                {
                    '函数名1': 函数1,
                    '函数名2': 函数2,
                    ...
                }
                其中每个函数必须接受(t, y)参数并返回导数向量
        
        功能:
            1. 将新函数添加到self.dynamics_funcs字典中
            2. 覆盖同名函数(如果存在)
            
        使用示例:
            def my_dynamics(t, y):
                # 自定义动力学实现
                return derivatives
            
            orbit.update_dynamics_funcs({
                'custom': my_dynamics
            })
            
        注意:
            添加的函数必须与_N_body_dynamics具有相同的签名:
            - 输入: (t, y)
            - 输出: 导数向量
        """
        self.dynamics_funcs.update(dynamics_funcs_dict)

    def rotation_matrix(self, now_frame, target_frame, time):
        """计算坐标系旋转矩阵
        使用SPICE计算两个坐标系之间的旋转矩阵
        
        参数:
            now_frame: 当前坐标系名称(SPICE ID格式字符串)，如:
                - 'ECLIPJ2000': J2000黄道坐标系
                - 'J2000': J2000赤道坐标系
                - 'IAU_EARTH': 地球固定坐标系
            target_frame: 目标坐标系名称(SPICE ID格式字符串)
            time: UTC时间字符串(格式如'2025 Jan 01 00:00:00')
            
        返回:
            3x3 numpy数组表示的旋转矩阵，可用于将向量从now_frame转换到target_frame:
                v_target = rotation_matrix @ v_now
                
        数学原理:
            旋转矩阵R满足:
            1. R^T = R^-1 (正交矩阵)
            2. det(R) = 1 (保持方向)
            3. 通过SPICE的pxform函数计算得到
            
        使用示例:
            # 从J2000赤道坐标系转换到黄道坐标系
            R = orbit.rotation_matrix('J2000', 'ECLIPJ2000', '2025 Jan 01 00:00:00')
            v_ecliptic = R @ v_equatorial
            
        注意:
            1. 时间参数会影响某些坐标系转换(如地球固定坐标系)
            2. 坐标系名称必须符合SPICE规范
            3. 返回的矩阵是正交矩阵，可直接用于向量变换
        """
        rotation_matrix = spice.pxform(now_frame, target_frame, spice.str2et(time))
        return rotation_matrix
    
    def predict(self, startyear, endyear,  
                dynamics_func=None,
                method=None, 
                rtol=None,
                atol=None, 
                stepsize=None,
            ):
        """
        预测天体轨道(使用数值积分)
        
        参数:
            startyear: 起始年份(整数)，如2025
            endyear: 结束年份(整数)，必须大于等于startyear
            dynamics_func: 使用的动力学函数名称(字符串)，可选值:
                - 'standard': 标准N体动力学(默认)
                - 其他自定义函数名(需先通过update_dynamics_funcs添加)
            method: 积分方法(字符串)，可选值:
                - 'RK45': 4/5阶Runge-Kutta(默认)
                - 'DOP853': 8阶Runge-Kutta
                - 其他scipy.integrate.solve_ivp支持的方法
            rtol: 相对误差容限(浮点数)，控制积分精度，默认1e-6
            atol: 绝对误差容限(浮点数)，控制积分精度，默认1e-6
            stepsize: 积分步长(秒, 整数)，默认600秒(10分钟)
            
        返回:
            生成器对象，每次yield一个元组包含:
            - time_str: UTC时间字符串(格式: 'YYYY Mon DD HH:MM:SS')
            - states: 天体状态数组(6*N维, 单位: km和km/s)，结构为:
                [x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, ...]
                
        数学原理:
            1. 使用solve_ivp求解常微分方程组:
                dy/dt = f(t,y)
            2. 动力学方程基于牛顿万有引力定律:
                F = G*m1*m2/r^2
                a = F/m
            3. 采用批处理方式积分以提高效率
            4. 转换为观测中心坐标系以简化分析
            
        工作流程:
            1. 从SPICE内核获取初始状态
            2. 将状态转换为米制单位
            3. 分批次进行数值积分(每批100步)
            4. 将结果转换回千米制单位
            5. 转换为观测中心坐标系
            6. 生成预测结果
            
        使用示例:
            # 预测2025-2030年轨道
            orbit = Orbit()
            for time_str, states in orbit.predict(2025, 2030):
                print(f"Time: {time_str}")
                print(f"Earth state: {states[0:6]}")
                
        注意:
            1. 长时间跨度预测会消耗较多内存
            2. 较小的rtol/atol值会提高精度但降低速度
            3. 结果已转换为观测中心坐标系
            4. 使用生成器模式避免内存爆炸
        """
        # 获取初始状态(只需要获取一次)
        start_time = f"{startyear} Jan 01 00:00:00"
        end_time = f"{endyear + 1} Jan 01 00:00:00"
        et_start = spice.str2et(start_time)
        diff_time = (parse_time(end_time) - parse_time(start_time)).total_seconds()
        if stepsize is None:
            stepsize = self.default_params['stepsize']
        # Get initial states for all bodies in self.bodies relative to observer
        states = []
        for body in self.bodies:
            state, _ = spice.spkezr(body, et_start, self.frame, "NONE", self.observer)
            states.append(state)
        y0 = np.concatenate(states) * 1000  # Convert to meters
        
        # 生成初始状态
        time_str = start_time
        states = y0 / 1000  # 转换回千米
        yield time_str, states
        
        # 数值积分计算轨道
        batch_size = 100  # 每100步处理一次
        batch_count = int(diff_time // stepsize) // batch_size
        remainder = int(diff_time // stepsize) % batch_size
        
        for batch in range(batch_count):
            # 计算100步的结果
            t_span = (0, stepsize * batch_size)
            t_eval = np.linspace(0, stepsize * batch_size, batch_size + 1)
            sol = self._my_solve_ivp(y0, t_span=t_span, t_eval=t_eval,
                                    method=method, rtol=rtol, atol=atol,
                                    dynamics_func=dynamics_func)
            
            # yield这100步的结果
            for i in range(1, len(sol.t)):  # 跳过初始状态(已yield)
                time_str = add_time(time_str, stepsize)
                states = sol.y.transpose()[i] / 1000  # 转换回千米
                yield time_str, states
            
            y0 = sol.y.transpose()[-1]
        
        # 处理剩余步数
        if remainder > 0:
            t_span = (0, stepsize * remainder)
            t_eval = np.linspace(0, stepsize* remainder, remainder + 1)
            sol = self._my_solve_ivp(y0, t_span=t_span, t_eval=t_eval,
                                    method=method, rtol=rtol, atol=atol,
                                    dynamics_func=dynamics_func)
            
            for i in range(1, len(sol.t)):
                time_str = add_time(time_str, stepsize)
                states = sol.y.transpose()[i] / 1000  # 转换回千米
                yield time_str, states
    
    def get_official_data(self, startyear, endyear, stepsize=None):
        """
        获取官方轨道数据(直接从SPICE内核获取)
        
        参数:
            startyear: 起始年份(整数)，如2025
            endyear: 结束年份(整数)，必须大于等于startyear
            stepsize: 时间步长(秒, 整数)，默认600秒(10分钟)
            
        返回:
            生成器对象，每次yield一个元组包含:
            - time_str: UTC时间字符串(格式: 'YYYY Mon DD HH:MM:SS')
            - states: 天体状态数组(6*N维, 单位: km和km/s)，结构为:
                [x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, ...]
                
        数据来源:
            直接从SPICE内核获取官方轨道数据，不经过数值积分计算
            
        工作流程:
            1. 计算时间范围内的所有时间点
            2. 对每个时间点:
               - 从SPICE获取各天体的状态
               - 转换为观测中心坐标系
               - 生成(time_str, states)元组
               
        使用示例:
            # 获取2025-2030年官方轨道数据
            orbit = Orbit()
            for time_str, states in orbit.get_official_data(2025, 2030):
                print(f"Time: {time_str}")
                print(f"Earth state: {states[0:6]}")
                
        注意:
            1. 数据直接从SPICE内核获取，精度最高
            2. 长时间跨度会消耗较多内存
            3. 使用生成器模式避免内存爆炸
            4. 结果已转换为观测中心坐标系
        """
        if stepsize is None:
            stepsize = self.default_params['stepsize']
        start_time = f"{startyear} Jan 01 00:00:00"
        end_time = f"{endyear + 1} Jan 01 00:00:00"
        diff_time = (parse_time(end_time) - parse_time(start_time)).total_seconds()
        n_timesteps = int(diff_time / stepsize) + 1
        et_start = spice.str2et(start_time)
        
        # 计算并yield轨道数据
        for i in range(n_timesteps):
            et = et_start + i * stepsize
            time_str = spice.et2utc(et, 'C', 0)
            states = []
            
            for body in self.bodies:
                state, _ = spice.spkezr(body, et, self.frame, 'NONE', self.observer)
                states.extend(state)
            
            yield time_str, states
    
    def _analyze_error(self, params):
        """
        分析单个参数组合的预测误差(内部方法)
        
        参数:
            params: 包含所有必要参数的字典，结构为:
                {
                    'startyear': 起始年份(整数),
                    'endyear': 结束年份(整数),
                    'dynamics_func': 动力学函数名称(字符串),
                    'method': 积分方法(字符串),
                    'rtol': 相对误差容限(浮点数),
                    'atol': 绝对误差容限(浮点数),
                    'stepsize': 积分步长(秒, 整数),
                    'param_key': 参数组合的唯一标识符(字符串)
                }
            
        返回:
            position_errors: 字典，包含各天体的位置误差数据
            velocity_errors: 字典，包含各天体的速度误差数据
            
        功能:
            1. 使用指定参数组合调用predict方法获取预测轨道
            2. 与官方数据比较计算误差
            3. 按月统计平均误差
            4. 返回该参数组合的误差数据
            
        误差计算方法:
            1. 位置误差: 预测位置与官方位置的欧氏距离(km)
            2. 速度误差: 预测速度与官方速度的欧氏距离(km/s)
            3. 按月计算平均值
            
        注意:
            1. 这是内部方法，不应直接调用
            2. 优化为只处理单个参数组合，提高并行处理能力
            3. 误差计算基于SPICE官方数据作为基准
        """
        # 解包参数
        startyear = params['startyear']
        endyear = params['endyear']
        dynamics_func = params['dynamics_func']
        method = params['method']
        rtol = params['rtol']
        atol = params['atol']
        stepsize = params['stepsize']
        
        position_errors = {body.lower(): [] for body in self.bodies}
        velocity_errors = {body.lower(): [] for body in self.bodies}
        
        current_month = None
        monthly_pos_errors = {body.lower(): [] for body in self.bodies}
        monthly_vel_errors = {body.lower(): [] for body in self.bodies}
        
        # 获取预测数据生成器
        predict_gen = self.predict(
            startyear=startyear,
            endyear=endyear,
            method=method,
            rtol=rtol,
            atol=atol,
            dynamics_func=dynamics_func,
            stepsize=stepsize,
        )
        
        for time_str, states in predict_gen:
            dt = parse_time(time_str)
            month_key = (dt.year, dt.month)
            et = spice.str2et(time_str)
            
            if month_key != current_month:
                if current_month is not None:
                    for body in self.bodies:
                        b = body.lower()
                        position_errors[b].append(np.mean(monthly_pos_errors[b]))
                        velocity_errors[b].append(np.mean(monthly_vel_errors[b]))
                
                current_month = month_key
                monthly_pos_errors = {body.lower(): [] for body in self.bodies}
                monthly_vel_errors = {body.lower(): [] for body in self.bodies}
            
            # 计算各天体误差
            for i, body in enumerate(self.bodies):
                b = body.lower()
                pos_start = i*6
                vel_start = i*6 + 3
                
                # 获取预测状态
                pred_pos = np.array(states[pos_start:pos_start+3])
                pred_vel = np.array(states[vel_start:vel_start+3])
                
                # 从SPICE获取官方状态
                official_state, _ = spice.spkezr(body, et, self.frame, "NONE", self.observer)
                official_pos = np.array(official_state[:3])
                official_vel = np.array(official_state[3:6])
                
                # 计算误差
                pos_error = np.linalg.norm(official_pos - pred_pos)
                vel_error = np.linalg.norm(official_vel - pred_vel)
                
                monthly_pos_errors[b].append(pos_error)
                monthly_vel_errors[b].append(vel_error)
        
        # 处理最后一个月数据
        if monthly_pos_errors[self.bodies[0].lower()]:
            for body in self.bodies:
                b = body.lower()
                position_errors[b].append(np.mean(monthly_pos_errors[b]))
                velocity_errors[b].append(np.mean(monthly_vel_errors[b]))
        
        return position_errors, velocity_errors
    
    def visualize_error(self, startyear, endyear, methods=None, rtols=None, atols=None, dynamics_funcs=None, stepsize=None):
        """
        可视化轨道预测误差
        
        参数:
            startyear: 起始年份(整数)，如2025
            endyear: 结束年份(整数)，必须大于等于startyear
            methods: 积分方法列表(字符串列表)，可选值:
                - ['RK45']: 4/5阶Runge-Kutta(默认)
                - ['DOP853']: 8阶Runge-Kutta
                - 其他scipy.integrate.solve_ivp支持的方法
                - None表示使用默认方法
            rtols: 相对误差容限列表(浮点数列表)，控制积分精度，默认[1e-6]
            atols: 绝对误差容限列表(浮点数列表)，控制积分精度，默认[1e-6]
            dynamics_funcs: 动力学函数列表(字符串列表)，可选值:
                - ['standard']: 标准N体动力学(默认)
                - 其他自定义函数名(需先通过update_dynamics_funcs添加)
                - None表示使用默认函数
            stepsize: 积分步长(秒, 整数)，默认600秒(10分钟)
            
        功能:
            1. 计算不同参数组合下的轨道预测误差
            2. 生成三种可视化图表:
               - average_errors.png: 各方法对各天体的平均误差柱状图
               - position_errors.png: 各天体位置误差随时间变化曲线
               - velocity_errors.png: 各天体速度误差随时间变化曲线
            3. 将图表保存到output/orbit_error_visualize目录
            
        可视化内容:
            1. 平均误差图:
                - 上子图: 位置误差(对数坐标)
                - 下子图: 速度误差(对数坐标)
                - 不同方法用不同颜色区分
            2. 位置/速度误差变化图:
                - 每个天体一个子图
                - 显示所有参数组合的误差曲线
                - 按月平均误差
            
        输出文件:
            保存到output/orbit_error_visualize目录下的PNG文件:
            1. average_errors.png
            2. position_errors.png 
            3. velocity_errors.png
            
        使用示例:
            # 可视化2025-2030年轨道误差
            orbit = Orbit()
            orbit.visualize(2025, 2030, 
                          methods=['RK45', 'DOP853'],
                          rtols=[1e-6, 1e-8],
                          atols=[1e-6, 1e-8])
            
        注意:
            1. 长时间跨度可视化会消耗较多计算资源
            2. 较小的rtol/atol值会提高精度但降低速度
            3. 图表会自动保存并显示在屏幕上
            4. 输出目录会自动创建
        """
        # 获取参数并解包为局部变量
        params = self._get_params(
            methods=methods,
            rtols=rtols,
            atols=atols,
            dynamics_funcs=dynamics_funcs,
            stepsize=stepsize
        )
        methods = params['methods']
        rtols = params['rtols']
        atols = params['atols']
        dynamics_funcs = params['dynamics_funcs']
        stepsize = params['stepsize']
        
        # 准备输出目录
        output_dir = os.path.join(self.output_dir, 'orbit_error_visualize')
        os.makedirs(output_dir, exist_ok=True)  
       
        # 遍历所有参数组合，计算误差
        all_pos_errors = {}
        all_vel_errors = {}
        param_labels = []
        
        # 显示进度条
        total_combinations = len(dynamics_funcs) * len(methods) * len(rtols) * len(atols)
        pbar = tqdm(total=total_combinations)
        
        # 将for循环移到visualize内部
        for dynamics_func in dynamics_funcs:
            for method in methods:
                for rtol in rtols:
                    for atol in atols:
                        param_key = f"{dynamics_func}_{method}_rtol{rtol}_atol{atol}"
                        param_labels.append(param_key)
                        
                        # 准备单个参数组合
                        params = {
                            'startyear': startyear,
                            'endyear': endyear,
                            'dynamics_func': dynamics_func,
                            'method': method,
                            'rtol': rtol,
                            'atol': atol,
                            'stepsize': stepsize,
                            'param_key': param_key
                        }
                        
                        # 分析单个参数组合的误差
                        all_pos_errors[param_key], all_vel_errors[param_key] = self._analyze_error(params) 
                        
                        pbar.update(1)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = plt.cm.tab20.colors
        
        # 1. 绘制所有星体位置误差图(每个星体一个子图)
        fig_pos, axes_pos = plt.subplots(len(self.bodies), 1, figsize=(12, 6*len(self.bodies)))
        if len(self.bodies) == 1:
            axes_pos = [axes_pos]
        
        for i, body in enumerate(self.bodies):
            b = body.lower()
            pos_errors = [np.mean(all_pos_errors[param_key][b]) for param_key in param_labels]
            
            x = np.arange(len(param_labels))
            axes_pos[i].bar(x, pos_errors, color=colors[:len(param_labels)])
            
            axes_pos[i].set_title(f'{body} Position Errors')
            axes_pos[i].set_ylabel('Avg Position Error (km)')
            axes_pos[i].set_xticks(x)
            axes_pos[i].set_xticklabels(param_labels, rotation=45, ha='right')
            axes_pos[i].grid(True, axis='y')
            axes_pos[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_position_errors.png'), 
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        # 2. 绘制所有星体速度误差图(每个星体一个子图)
        fig_vel, axes_vel = plt.subplots(len(self.bodies), 1, figsize=(12, 6*len(self.bodies)))
        if len(self.bodies) == 1:
            axes_vel = [axes_vel]
        
        for i, body in enumerate(self.bodies):
            b = body.lower()
            vel_errors = [np.mean(all_vel_errors[param_key][b]) for param_key in param_labels]
            
            x = np.arange(len(param_labels))
            axes_vel[i].bar(x, vel_errors, color=colors[:len(param_labels)])
            
            axes_vel[i].set_title(f'{body} Velocity Errors')
            axes_vel[i].set_ylabel('Avg Velocity Error (km/s)')
            axes_vel[i].set_xticks(x)
            axes_vel[i].set_xticklabels(param_labels, rotation=45, ha='right')
            axes_vel[i].grid(True, axis='y')
            axes_vel[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_velocity_errors.png'), 
                   bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

        # 3. 绘制位置误差折线图
        fig, axes = plt.subplots(len(self.bodies), 1, figsize=(15, 5*len(self.bodies)))
        fig.suptitle(f'Average Orbit Errors ({startyear}-{endyear})\n'
                     f'{', '.join(self.bodies)}\n'
                     f'Dynamics: {", ".join(dynamics_funcs)} | Method: {", ".join(methods)}\nRTOLs: {", ".join(map(str, rtols))} | ATOLs: {", ".join(map(str, atols))}', 
                    y=1.05)
        
        for i, body in enumerate(self.bodies):
            ax = axes[i]
            b = body.lower()
            for j, param_key in enumerate(param_labels):
                ax.plot(all_pos_errors[param_key][b], 
                       label=param_key, 
                       color=colors[j % len(colors)],
                       alpha=0.8)
            
            ax.set_title(f'{body} Position Error')
            ax.set_xlabel('Month')
            ax.set_ylabel('Error (km)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_errors.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()

        # 4. 绘制速度误差折线图
        fig, axes = plt.subplots(len(self.bodies), 1, figsize=(15, 5*len(self.bodies)))
        fig.suptitle(f'Average Orbit Errors ({startyear}-{endyear})\n'
                     f'{', '.join(self.bodies)}\n'
                     f'Dynamics: {", ".join(dynamics_funcs)} | Method: {", ".join(methods)}\nRTOLs: {", ".join(map(str, rtols))} | ATOLs: {", ".join(map(str, atols))}', 
                    y=1.05)
        
        for i, body in enumerate(self.bodies):
            ax = axes[i]
            b = body.lower()
            for j, param_key in enumerate(param_labels):
                ax.plot(all_vel_errors[param_key][b], 
                       label=param_key, 
                       color=colors[j % len(colors)],
                       alpha=0.8)
            
            ax.set_title(f'{body} Velocity Error')
            ax.set_xlabel('Month')
            ax.set_ylabel('Error (km/s)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'velocity_errors.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()
        
        # 5. 输出CSV文件
        csv_path = os.path.join(output_dir, 'error_data.csv')
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            # 写入表头
            headers = ['param_key', 'body', 'avg_position_error', 'avg_velocity_error']
            writer.writerow(headers)
            
            # 写入数据
            for param_key in param_labels:
                for body in self.bodies:
                    b = body.lower()
                    row = [
                        param_key,
                        body,
                        np.mean(all_pos_errors[param_key][b]),
                        np.mean(all_vel_errors[param_key][b])
                    ]
                    writer.writerow(row)

if __name__=='__main__':
    orbit = Orbit(
        bodies=['EARTH', 'MOON', 'SUN', 'JUPITER'], 
        observer='SUN', 
        frame='ECLIPJ2000', 
        masses={ 'EARTH': 5.972e+24,'MOON': 7.342e+22,'SUN': 1.989e+30, 'JUPITER':1.898e+27}, 
        kernel_files=['data/de442.bsp', 'data/earth_200101_990827_predict.bpc', 'data/naif0012.tls', 'data/jup346.bsp'], 
        output_dir='output'
    )
    orbit.visualize(startyear=2025,
                endyear=2075,
                methods=['RK45', 'DOP853'],
                rtols=[1e-6, 1e-8],
                atols=[1e-6, 1e-8],
                dynamics_funcs=['standard'],
                stepsize=1200)
