# AQI Prediction System

A machine learning-based system to predict Air Quality Index (AQI) from pollutant concentrations in the environment. This project uses a Random Forest regression model trained on real-world air quality data from multiple Indian cities to accurately predict AQI values based on various pollutant measurements.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [AQI Categories](#aqi-categories)
- [Project Structure](#project-structure)
- [Example Output](#example-output)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Features

### Core Functionality

- **Machine Learning Model**: Random Forest Regressor trained on 24,850+ real-world air quality records
- **High Accuracy**: Achieves 90.72% R² score on test data with MAE of 20.90
- **12 Pollutant Support**: Predicts AQI from PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, and Xylene
- **Flexible Input**: Accepts any combination of pollutants - missing values are automatically imputed
- **AQI Categorization**: Automatically categorizes predictions into 6 levels (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

### CLI Commands

- **`predict`**: Single AQI prediction from command-line pollutant values
  - Supports all 12 pollutants as optional flags
  - JSON output option (`--json`)
  - Save results to file (`--output`)
  - Custom model path support (`--model`)
  
- **`interactive`**: User-friendly interactive mode
  - Step-by-step prompts for pollutant input
  - Input validation with error messages
  - Skip pollutants by pressing Enter
  - Real-time validation feedback
  
- **`batch`**: Process multiple records from CSV files
  - CSV and JSON output formats
  - Statistics display (`--show-stats`)
  - Category distribution analysis
  - Handles missing columns automatically
  
- **`model-info`**: Display comprehensive model information
  - Model type and hyperparameters
  - Feature importance rankings with visual bars
  - Median values used for imputation
  - Complete feature list
  
- **`info`**: System documentation and help
  - Pollutant descriptions and units
  - AQI category explanations
  - Usage examples
  - Command reference

### Data Processing Features

- **Missing Value Imputation**: Automatically fills missing pollutants with median values from training data
- **Input Normalization**: Handles various input formats (case-insensitive, different naming conventions)
- **Data Validation**: Prevents negative values and validates input types
- **Feature Mapping**: Intelligent mapping between CLI flags and model features

### Output Features

- **Multiple Formats**: 
  - Formatted text output with color coding
  - JSON format for programmatic access
  - CSV format for spreadsheet analysis
  
- **Color-coded Display**: Visual indicators for AQI categories
  - Green (Good), Yellow (Satisfactory), Bright Yellow (Moderate)
  - Red (Poor), Magenta (Very Poor), Bright Red (Severe)
  
- **Health Recommendations**: Contextual health advice based on AQI category
- **Detailed Reports**: Shows input pollutants, imputed values, and predictions
- **Statistics**: Batch processing includes mean, median, min, max, std dev, and category distribution
- **Timestamp**: JSON output includes ISO timestamp for tracking

### Model Features

- **Feature Importance**: Identifies most critical pollutants (PM2.5: 49.92%, CO: 37.30%)
- **Model Persistence**: Saves trained model with metadata for easy loading
- **Reproducible**: Fixed random seed ensures consistent results
- **Efficient Training**: Uses parallel processing (`n_jobs=-1`) for faster training
- **Comprehensive Metrics**: Displays MAE, RMSE, and R² for training and test sets

### User Experience Features

- **Error Handling**: Comprehensive error messages with helpful suggestions
- **Progress Indicators**: Visual feedback during batch processing
- **Help System**: Built-in help for all commands (`--help`)
- **Example Usage**: Inline examples in error messages
- **Flexible Configuration**: Custom model paths and output locations

### Technical Features

- **Clean Code**: Well-structured, maintainable codebase
- **Type Safety**: Input validation and type checking
- **Documentation**: Comprehensive docstrings and comments
- **Modular Design**: Reusable functions and components
- **Cross-platform**: Works on Windows, macOS, and Linux

## System Requirements

- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for training)
- 500MB free disk space for datasets and models
- Operating System: Windows, macOS, or Linux

## Installation

### Step 1: Clone or Download the Repository

```bash
cd ml_project
```

### Step 2: Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas==2.1.4` - Data manipulation and analysis
- `numpy==1.26.2` - Numerical computing
- `scikit-learn==1.4.0` - Machine learning library
- `joblib==1.3.2` - Model serialization
- `click==8.1.7` - CLI framework

### Step 3: Verify Installation

```bash
python3 --version
pip list | grep -E "pandas|numpy|scikit-learn|click|joblib"
```

If all packages are listed, installation is successful.

## Dataset Information

The model is trained on air quality data from **26 Indian cities** collected between **2015-01-01** and **2020-07-01**.

### Dataset Statistics

- **Total Records**: 29,531 daily measurements
- **Records with AQI**: 24,850 (used for training)
- **Cities Covered**: 26 cities including Ahmedabad, Bengaluru, Chennai, Delhi, Mumbai, and more
- **Time Period**: January 2015 to July 2020
- **Features**: 12 pollutant measurements

### Pollutants Tracked

1. **PM2.5** (Particulate Matter 2.5) - Fine particles, μg/m³
2. **PM10** (Particulate Matter 10) - Coarse particles, μg/m³
3. **NO** (Nitrogen Oxide) - μg/m³
4. **NO2** (Nitrogen Dioxide) - μg/m³
5. **NOx** (Nitrogen Oxides) - μg/m³
6. **NH3** (Ammonia) - μg/m³
7. **CO** (Carbon Monoxide) - mg/m³
8. **SO2** (Sulfur Dioxide) - μg/m³
9. **O3** (Ozone) - μg/m³
10. **Benzene** - Volatile organic compound, μg/m³
11. **Toluene** - Volatile organic compound, μg/m³
12. **Xylene** - Volatile organic compound, μg/m³

### Data Files

- `city_day.csv` - Daily aggregated data by city (used for training)
- `stations.csv` - Station metadata

## Usage

### Step 1: Train the Model

First, train the model using the provided dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the data from `data/city_day.csv`
- Handle missing values using median imputation
- Split data into training (80%) and testing (20%) sets
- Train a Random Forest regression model with 100 estimators
- Evaluate model performance using MAE, RMSE, and R² metrics
- Save the model to `models/aqi_model.pkl` along with feature list and median values
- Display training metrics, test metrics, and feature importance

**Expected Training Time**: 1-3 minutes depending on system performance

**Output**: The model file `models/aqi_model.pkl` will be created in the `models/` directory.

### Step 2: Predict AQI

#### Single Prediction

Predict AQI from pollutant values:

```bash
python predict_aqi.py predict --pm25 50 --no2 30 --co 1.5 --so2 25 --o3 60
```

You can provide any combination of pollutants:
- `--pm25`: PM2.5 concentration (μg/m³)
- `--pm10`: PM10 concentration (μg/m³)
- `--no`: NO concentration (μg/m³)
- `--no2`: NO2 concentration (μg/m³)
- `--nox`: NOx concentration (μg/m³)
- `--nh3`: NH3 concentration (μg/m³)
- `--co`: CO concentration (mg/m³)
- `--so2`: SO2 concentration (μg/m³)
- `--o3`: O3 concentration (μg/m³)
- `--benzene`: Benzene concentration (μg/m³)
- `--toluene`: Toluene concentration (μg/m³)
- `--xylene`: Xylene concentration (μg/m³)

**Additional options:**
- `--json`: Output results in JSON format
- `--output FILE`: Save results to a file (CSV or JSON)

**Examples:**
```bash
# JSON output
python predict_aqi.py predict --pm25 50 --no2 30 --json

# Save to file
python predict_aqi.py predict --pm25 50 --no2 30 --output result.json
```

#### Interactive Mode

For easier input, use interactive mode:

```bash
python predict_aqi.py interactive
```

This will prompt you to enter pollutant values one by one. Press Enter to skip any pollutant.

#### Batch Prediction

Predict AQI for multiple records from a CSV file:

```bash
python predict_aqi.py batch --file input_data.csv --output predictions.csv
```

**Options:**
- `--file`: Input CSV file (required)
- `--output`: Output file path (default: predictions.csv)
- `--format`: Output format - csv or json (default: csv)
- `--show-stats`: Display statistics about predictions

**Examples:**
```bash
# Basic batch prediction
python predict_aqi.py batch --file data.csv

# With statistics
python predict_aqi.py batch --file data.csv --show-stats

# JSON output
python predict_aqi.py batch --file data.csv --format json --output results.json
```

The input CSV should contain columns with pollutant names (PM2.5, PM10, NO2, etc.). Missing values will be filled with median values from the training data.

#### Model Information

View details about the trained model:

```bash
python predict_aqi.py model-info
```

This displays:
- Model type and parameters
- Feature list
- Feature importance rankings
- Median values used for imputation

#### Display Information

Get information about the system:

```bash
python predict_aqi.py info
```

## AQI Categories

- **Good (0-50)**: Minimal impact
- **Satisfactory (51-100)**: Minor breathing discomfort
- **Moderate (101-200)**: Breathing discomfort to people with lung disease
- **Poor (201-300)**: Breathing discomfort to most people
- **Very Poor (301-400)**: Respiratory illness on prolonged exposure
- **Severe (401+)**: Health hazard

## Project Structure

```
ml_project/
├── data/                    # Dataset files
│   ├── city_day.csv        # Daily aggregated data by city (primary training data)
│   ├── city_hour.csv       # Hourly data by city
│   ├── station_day.csv     # Daily data by monitoring station
│   ├── station_hour.csv    # Hourly data by monitoring station
│   └── stations.csv        # Station metadata and locations
├── models/                  # Trained models (created after training)
│   └── aqi_model.pkl      # Serialized Random Forest model with metadata
├── train_model.py          # Model training script
│   ├── load_and_preprocess_data()  # Data loading and preprocessing
│   ├── train_model()              # Model training and evaluation
│   └── save_model()               # Model persistence
├── predict_aqi.py          # CLI prediction application
│   ├── load_model()              # Load trained model
│   ├── make_prediction()         # Core prediction function
│   ├── categorize_aqi()         # AQI category mapping
│   ├── get_health_recommendation() # Health advice generator
│   ├── predict command           # Single prediction CLI
│   ├── interactive command       # Interactive mode CLI
│   ├── batch command             # Batch processing CLI
│   ├── model-info command        # Model information CLI
│   └── info command              # System information CLI
├── requirements.txt        # Python dependencies with versions
├── README.md               # Comprehensive documentation (this file)
├── QUICK_START.md          # Quick reference guide
└── .gitignore             # Git ignore patterns
```

### File Descriptions

- **train_model.py**: Script to train the Random Forest model on the dataset
- **predict_aqi.py**: Main CLI application with all prediction commands
- **requirements.txt**: Python package dependencies
- **README.md**: Complete project documentation
- **QUICK_START.md**: Quick reference for common commands
- **.gitignore**: Excludes model files and Python cache from version control

## Example Output

### Single Prediction

```
======================================================================
AQI Prediction Results
======================================================================

Predicted AQI: 156.23
AQI Category: Moderate

Health Recommendation:
  People with lung disease, children, and older adults should reduce prolonged outdoor exertion.

Input Pollutants:
  • PM2.5: 50.0 μg/m³
  • NO2: 30.0 μg/m³
  • CO: 1.5 mg/m³

Imputed Pollutants (using median values):
  • PM10: 100.00 μg/m³
  • SO2: 30.00 μg/m³
======================================================================
```

### Batch Prediction with Statistics

```
✓ Predictions completed!
Total records: 100
Results saved to: predictions.csv

======================================================================
Prediction Statistics
======================================================================

AQI Statistics:
  Mean: 145.23
  Median: 142.50
  Min: 85.30
  Max: 320.45
  Std Dev: 45.67

Category Distribution:
  Good: 15 (15.0%)
  Satisfactory: 25 (25.0%)
  Moderate: 35 (35.0%)
  Poor: 20 (20.0%)
  Very Poor: 5 (5.0%)
```

## Model Performance

The trained model achieves the following performance metrics:

### Training Metrics
- **Mean Absolute Error (MAE)**: ~11.30
- **Root Mean Squared Error (RMSE)**: ~24.26
- **R² Score**: 0.9708 (97.08% variance explained)

### Test Metrics
- **Mean Absolute Error (MAE)**: ~20.90
- **Root Mean Squared Error (RMSE)**: ~41.23
- **R² Score**: 0.9072 (90.72% variance explained)

### Feature Importance

The model identifies the most important pollutants for AQI prediction:

1. **PM2.5** (49.92%) - Most critical feature
2. **CO** (37.30%) - Second most important
3. **PM10** (3.59%)
4. **NO** (3.55%)
5. **O3** (1.19%)
6. **NOx** (1.06%)
7. **SO2** (0.85%)
8. **NO2** (0.71%)
9. **Toluene** (0.62%)
10. **Xylene** (0.51%)

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Number of Trees**: 100 estimators
- **Max Depth**: 20 levels
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Random State**: 42 (for reproducibility)

## Technical Details

### Data Preprocessing

1. **Missing Value Handling**: Median imputation for each pollutant feature
2. **Feature Selection**: 12 pollutant features used as input
3. **Target Variable**: AQI (continuous numeric value)
4. **Data Split**: 80% training, 20% testing with random state 42

### Prediction Pipeline

1. **Input Validation**: Checks for negative values and required inputs
2. **Feature Normalization**: Maps input keys to standardized feature names
3. **Missing Value Imputation**: Uses median values from training data
4. **Prediction**: Random Forest model generates AQI value
5. **Categorization**: Maps numeric AQI to category (Good, Satisfactory, etc.)
6. **Output Formatting**: Displays results with color coding and health recommendations

### Model Persistence

The trained model is saved using `joblib` and includes:
- Trained Random Forest model object
- Feature list (12 pollutants)
- Median values for each feature (used for imputation)

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error

**Error**: `Model not found at models/aqi_model.pkl`

**Solution**: 
```bash
python train_model.py
```

#### 2. Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. Permission Denied

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**: 
- On Linux/Mac: Use `sudo` or check file permissions
- On Windows: Run terminal as Administrator

#### 4. Memory Error During Training

**Error**: `MemoryError` or system becomes slow

**Solution**:
- Close other applications
- Reduce dataset size for testing
- Use a machine with more RAM

#### 5. Invalid Pollutant Values

**Error**: `Value cannot be negative`

**Solution**: Ensure all pollutant values are non-negative numbers

#### 6. CSV File Not Found (Batch Mode)

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: 
- Check file path is correct
- Use absolute path if relative path doesn't work
- Ensure CSV file exists and is readable

#### 7. Empty Predictions

**Error**: `No pollutant values provided`

**Solution**: Provide at least one pollutant value using command-line flags or interactive mode

### Getting Help

If you encounter issues not listed here:

1. Check that all dependencies are installed correctly
2. Verify Python version is 3.7 or higher
3. Ensure data files are in the `data/` directory
4. Review error messages for specific guidance

## Notes

- The model uses median imputation for missing pollutant values
- At least one pollutant value must be provided for prediction
- The model is trained on daily aggregated data from multiple cities
- For best results, provide as many pollutant values as available
- PM2.5 and CO are the most important features for accurate predictions
- The model performs best when key pollutants (PM2.5, CO, PM10) are provided
- Predictions are based on daily averages, not real-time measurements

## Advanced Usage

### Custom Model Path

Use a different model file:

```bash
python predict_aqi.py predict --pm25 50 --no2 30 --model path/to/custom_model.pkl
```

### Batch Processing with Custom Format

```bash
# JSON output with statistics
python predict_aqi.py batch --file data.csv --format json --output results.json --show-stats

# CSV output with statistics
python predict_aqi.py batch --file data.csv --format csv --output results.csv --show-stats
```

### Integration with Scripts

You can integrate the CLI into your own scripts:

```python
import subprocess
import json

# Run prediction and capture JSON output
result = subprocess.run(
    ['python', 'predict_aqi.py', 'predict', '--pm25', '50', '--no2', '30', '--json'],
    capture_output=True,
    text=True
)
data = json.loads(result.stdout)
print(f"AQI: {data['predicted_aqi']}")
```

### CSV Input Format for Batch Processing

Your input CSV should have columns matching pollutant names:

```csv
PM2.5,PM10,NO2,CO,SO2,O3
50,100,30,1.5,25,60
75,120,45,2.0,30,70
```

Missing columns will be filled with median values automatically.

## Future Enhancements

- **GUI Interface**: Web-based or desktop GUI for easier interaction
- **Real-time Prediction API**: REST API for integration with other applications
- **Model Retraining**: Automated retraining with new data
- **Support for More Pollutants**: Additional pollutant types
- **Time Series Prediction**: Forecast future AQI values
- **Location-based Predictions**: City-specific model variants
- **Mobile App**: iOS/Android application
- **Data Visualization**: Charts and graphs for predictions
- **Model Comparison**: Compare different ML algorithms
- **Export Formats**: PDF reports, Excel files
- **Alert System**: Notifications for poor air quality
- **Historical Analysis**: Trend analysis and pattern recognition

## Contributing

Contributions are welcome! Areas for improvement:

1. Model performance optimization
2. Additional features and pollutants
3. Code optimization and refactoring
4. Documentation improvements
5. Test coverage
6. Bug fixes

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Dataset: Air quality data from Indian cities (2015-2020)
- Libraries: pandas, scikit-learn, numpy, click, joblib
- Model: Random Forest Regressor from scikit-learn

## Citation

If you use this project in your research, please cite:

```
AQI Prediction System
Machine Learning-based Air Quality Index Prediction
Trained on Indian city air quality data (2015-2020)
```

## Contact & Support

For issues, questions, or contributions:
- Check the [Troubleshooting](#troubleshooting) section
- Review example outputs and documentation
- Verify all dependencies are correctly installed

---

**Version**: 1.0.0  
**Last Updated**: December 2025  
**Python Version**: 3.7+

