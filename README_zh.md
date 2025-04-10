# 日食月食预测模型
**English** | [中文](README_zh.md)

一个基于Python的天文模型，用于预测和分析日食与月食。本项目提供n体轨道预测、精确食象预测、误差分析以及食象路径可视化工具。

## 演示
**更多信息请查看demo.ipynb**
- **预测食象轨道**
  - **日食**
<video src="https://github.com/user-attachments/assets/5298cb8f-43c7-4afa-80c6-99ddf207ccd9" controls width="600"></video>

  - **月食**
<video src="https://github.com/user-attachments/assets/6320a26e-53b1-4854-815d-ebec4636010f" controls width="600"></video>

- **预测轨道误差**
<div align="center"> 
  <img src="https://github.com/user-attachments/assets/402f6f70-9f86-445d-8e43-80c08e6943f8" alt="轨道误差" width="600">
  </div>

- **预测食象误差**
<div align="center">
  <img src="https://github.com/user-attachments/assets/e786eaae-e0c1-4703-8427-0596cb9b613e" alt="食象误差" width="600">
  </div> 

## 功能特性

- **N体轨道预测**
  - 实现n体引力动力学
  - 支持任意数量天体
  - 多种数值积分方法(RK45, DOP853等)
  - 高精度位置和速度计算

- **食象预测**
  - 日食和月食预测
  - 使用SPICE内核进行高精度天文计算
  - 支持多种数值积分方法
  - 可配置误差容限设置

- **分析与可视化**
  - 与参考数据对比的误差分析
  - 全球地图上的食象路径可视化
  - 食象动画视频生成
  - 多种指标的统计误差分析
  - 食象期间轨道可视化

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/cflyuke/LunaSol.git
cd LunaSol
```

2. 安装依赖项并以开发模式安装包:
```bash
pip install -r requirements.txt
pip install -e .
```

这将以开发模式安装包，允许您从Python环境的任何位置导入它。

## 项目结构
```
.
├── README.md
├── README_CN.md
├── demo.ipynb           # 在Jupyter笔记本中展示结果
├── LunaSol/             # 源代码
│   ├── Eclipse.py            # 基础食象计算类
│   ├── LunarEclipse.py       # 月食特定计算
│   ├── Orbit.py              # 轨道力学计算
│   ├── SolarEclipse.py       # 日食特定计算
│   └── utils.py              # 实用函数
├── data/                      # SPICE内核和参考数据
│   ├── de442.bsp            # JPL星历数据
│   ├── earth_200101_990827_predict.bpc
│   ├── jup346.bsp
│   ├── naif0012.tls
│   ├── lunar_eclipse_data.csv  # NASA标准食象预测
│   └── solar_eclipse_data.csv
├── result
│   ├── eclipse_video        # 食象轨道可视化
│   ├── orbit_error_visualize  # 轨道误差可视化
│   ├── output_3_bodies        # 仅考虑地球、月球、太阳的预测结果
│   └── output_with_JUPITER_VENUS # 考虑金星和木星的预测结果
├── requirements.txt
└── setup.py
```
数据参考: (https://naif.jpl.nasa.gov/pub/naif/generic_kernel)
日月食预测数据: (https://eclipse.gsfc.nasa.gov/)
## 依赖项

- numpy
- spiceypy
- pandas
- matplotlib
- cartopy
- opencv-python
- scipy
- tqdm

## 使用方法

### 基础食象预测

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
   - 食象时间和类型
   - 天体的位置和速度向量
   - 误差分析结果

2. **可视化**
   - 误差类型分布图
   - 时间误差分析图
   - 全球食象路径图
   - 食象进程动画视频

3. **误差分析**
   - 遗漏/错误/额外统计
   - 时间精度测量
   - 不同方法间的对比分析

## 详细功能

### 食象预测
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
- 全球食象路径映射
- 食象进程动画
- 误差分布图
- 食象期间轨道可视化

## 贡献

欢迎贡献！请随时提交Pull Request。
