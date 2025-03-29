# 日食月食预测模型

[English](README.md) | **中文**

一个基于Python的天文模型，用于预测和分析日食与月食。本项目提供n体轨道预测、精确日食月食预测、误差分析以及日食路径可视化工具。

## 演示
**在demo.ipynb中你能看到更多**
- **预测日/月食轨迹**
  - **日食**
<video src="https://github.com/user-attachments/assets/5298cb8f-43c7-4afa-80c6-99ddf207ccd9" controls width="600"></video>

  - **月食**
<video src="https://github.com/user-attachments/assets/6320a26e-53b1-4854-815d-ebec4636010f" controls width="600"></video>

- **预测轨道误差**
<div align="center"> 
  <img src="https://github.com/user-attachments/assets/ef4b573d-b1e1-4788-89fc-b2ca4a0755ab" alt="轨道误差" width="600">
  </div>

- **预测日食误差**
<div align="center">
  <img src="https://github.com/user-attachments/assets/55fd6fbf-0d0f-4d73-8898-04b330688f7e" alt="日食误差" width="600">
  </div> 


## 功能

- **N体轨道预测**
  - 实现n体引力动力学
  - 支持任意数量天体
  - 多种数值积分方法(RK45, DOP853等)
  - 高精度位置和速度计算

- **日食月食预测**
  - 日食和月食预测
  - 使用SPICE内核进行高精度天文计算
  - 支持多种数值积分方法
  - 可配置误差容限设置

- **分析与可视化**
  - 与参考数据对比的误差分析
  - 全球地图上的日食路径可视化
  - 生成日食动画视频
  - 使用多种指标的统计误差分析
  - 日食期间的轨道可视化

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/cflyuke/LunaSol.git
cd LunaSol
```

2. 安装依赖并开发模式安装包:
```bash
pip install -r requirements.txt
pip install -e .
```

这将以开发模式安装包，允许您从Python环境的任何位置导入它。

## 项目结构

```
.
├── README.md
├── README_zh.md
├── demo.ipynb
├── setup.py
├── LunaSol/             # 源代码
│   ├── Eclipse.py            # 基础日食计算类
│   ├── LunarEclipse.py       # 月食特定计算
│   ├── Orbit.py              # 轨道力学计算
│   ├── SolarEclipse.py       # 日食特定计算
│   └── utils.py              # 工具函数
├── data/                      # SPICE内核和参考数据
│   ├── de442.bsp            # JPL星历数据
│   ├── earth_200101_990827_predict.bpc
│   ├── jup346.bsp
│   ├── naif0012.tls
│   ├── lunar_eclipse_data.csv
│   └── solar_eclipse_data.csv
└── output/                  # 所有输出文件
    ├── Lunar_eclipse/
    ├── Lunar_eclipse_error_visualize/
    ├── Lunar_eclipse_video/
    ├── Solar_eclipse/
    ├── Solar_eclipse_error_visualize/
    ├── Solar_eclipse_video/
    └── orbit_error_visualize/
```
[数据参考]((https://naif.jpl.nasa.gov/pub/naif/generic_kernels))
## 依赖

- numpy
- spiceypy
- pandas
- matplotlib
- cartopy
- opencv-python
- scipy
- tqdm

## 使用

### 基础日食预测

```python
from LunaSol.SolarEclipse import SolarEclipse
from LunaSol.LunarEclipse import LunarEclipse

# 初始化日食预测器
solar = SolarEclipse()

# 预测2024-2025年的日食
results = solar.predict_eclipse(2024, 2025)

# 分析并可视化结果
solar.visualize_error(2024, 2025, results)
solar.visualize_orbit(2024, results[0])
```

### 完整分析流程

```python
# 运行完整分析流程
solar.pipeline(
    startyear=2024,
    endyear=2025,
    methods=['RK45', 'DOP853'],  # 积分方法
    rtols=[1e-6],                # 相对容差
    atols=[1e-6]                 # 绝对容差
)
```

## 输出

模型生成多种类型的输出:

1. **预测数据** (CSV格式)
   - 日食时间和类型
   - 天体的位置和速度向量
   - 误差分析结果

2. **可视化**
   - 误差类型分布图
   - 时间误差分析图
   - 全球日食路径地图
   - 日食进程动画视频

3. **误差分析**
   - 遗漏/错误/额外统计
   - 时间精度测量
   - 不同方法间的对比分析

## 详细功能

### 日食预测
- 支持日食和月食预测
- 使用高精度JPL星历数据
- 实现多种数值积分方法
- 可配置精度设置

### 误差分析
- 与参考数据对比预测结果
- 计算时间误差
- 分析预测准确性
- 生成统计可视化

### 可视化
- 日食路径的全球地图
- 日食进程动画
- 误差分布图
- 日食期间的轨道可视化

## 贡献

欢迎贡献！请随时提交Pull Request。
