# Eclipse Prediction Model
**English** | [中文](README_zh.md)

A Python-based astronomical model for predicting and analyzing solar and lunar eclipses. This project provides tools for n-stars orbit prediction, accurate eclipse predictions, error analysis, and visualization of eclipse paths.

## Demo
**You can get more information in demo.ipynb**
- **Predict Eclipse Orbit**
  - **Solar ecliipse**
<video src="https://github.com/user-attachments/assets/5298cb8f-43c7-4afa-80c6-99ddf207ccd9" controls width="600"></video>

  - **Lunar Eclipse**
<video src="https://github.com/user-attachments/assets/6320a26e-53b1-4854-815d-ebec4636010f" controls width="600"></video>

- **Predict Orbit Error**
<div align="center"> 
  <img src="https://github.com/user-attachments/assets/402f6f70-9f86-445d-8e43-80c08e6943f8" alt="orbit errors" width="600">
  </div>

- **Predict Eclipse Error**
<div align="center">
  <img src="https://github.com/user-attachments/assets/e786eaae-e0c1-4703-8427-0596cb9b613e" alt="Eclipse error" width="600">
  </div> 



## Features

- **N-body Orbital Prediction**
  - Implementation of n-body gravitational dynamics
  - Support for arbitrary number of celestial bodies
  - Multiple numerical integration methods (RK45, DOP853, etc.)
  - High-precision position and velocity calculations

- **Eclipse Prediction**
  - Solar and lunar eclipse prediction
  - High-precision astronomical calculations using SPICE kernels
  - Support for multiple numerical integration methods
  - Configurable error tolerance settings

- **Analysis & Visualization**
  - Error analysis comparing predictions with reference data
  - Visualization of eclipse paths on global maps
  - Generation of eclipse animation videos
  - Statistical error analysis with various metrics
  - Orbit visualization during eclipse periods

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cflyuke/LunaSol.git
cd LunaSol
```

2. Install required dependencies and the package in development mode:
```bash
pip install -r requirements.txt
pip install -e .
```

This will install the package in development mode, allowing you to import it from anywhere in your Python environment.

## Project Structure

```
.
├── README.md
├── README_CN.md
├── demo.ipynb
├── setup.py
├── LunaSol/             # Source code
│   ├── Eclipse.py            # Base eclipse calculation class
│   ├── LunarEclipse.py       # Lunar eclipse specific calculations
│   ├── Orbit.py              # Orbital mechanics calculations
│   ├── SolarEclipse.py       # Solar eclipse specific calculations
│   └── utils.py              # Utility functions
├── data/                      # SPICE kernels and reference data
│   ├── de442.bsp            # JPL ephemeris data
│   ├── earth_200101_990827_predict.bpc
│   ├── jup346.bsp
│   ├── naif0012.tls
│   ├── lunar_eclipse_data.csv
│   └── solar_eclipse_data.csv
└── output/                  # All output files
    ├── Lunar_eclipse/
    ├── Lunar_eclipse_error_visualize/
    ├── Lunar_eclipse_video/
    ├── Solar_eclipse/
    ├── Solar_eclipse_error_visualize/
    ├── Solar_eclipse_video/
    └── orbit_error_visualize/
```
[Data reference]((https://naif.jpl.nasa.gov/pub/naif/generic_kernels))
## Dependencies

- numpy
- spiceypy
- pandas
- matplotlib
- cartopy
- opencv-python
- scipy
- tqdm

## Usage

### Basic Eclipse Prediction

```python
from LunaSol.SolarEclipse import SolarEclipse
from LunaSol.LunarEclipse import LunarEclipse

# Initialize Solar Eclipse predictor
solar = SolarEclipse()

# Predict eclipses between 2024-2025
results = solar.predict_eclipse(2024, 2025)

# Analyze and visualize results
solar.visualize_error(2024, 2025, results)
solar.visualize_orbit(2024, results[0])
```

### Pipeline Execution

```python
# Run complete analysis pipeline
solar.pipeline(
    startyear=2024,
    endyear=2025,
    methods=['RK45', 'DOP853'],  # Integration methods
    rtols=[1e-6],                # Relative tolerances
    atols=[1e-6]                 # Absolute tolerances
)
```

## Output

The model generates several types of output:

1. **Prediction Data** (CSV format)
   - Eclipse timing and type
   - Position and velocity vectors for celestial bodies
   - Error analysis results

2. **Visualizations**
   - Error type distribution plots
   - Time error analysis graphs
   - Global eclipse path maps
   - Animation videos of eclipse progression

3. **Error Analysis**
   - Miss/mistake/addition statistics
   - Timing accuracy measurements
   - Comparative analysis between different methods

## Features in Detail

### Eclipse Prediction
- Supports both solar and lunar eclipse predictions
- Uses high-precision JPL ephemeris data
- Implements multiple numerical integration methods
- Configurable precision settings

### Error Analysis
- Compares predictions with reference data
- Calculates timing errors
- Analyzes prediction accuracy
- Generates statistical visualizations

### Visualization
- Global mapping of eclipse paths
- Animation of eclipse progression
- Error distribution plots
- Orbital visualization during eclipses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
