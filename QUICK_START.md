# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training the Model

```bash
python train_model.py
```

This will create `models/aqi_model.pkl` with the trained model.

## Using the CLI

### 1. Single Prediction (Command-line)

```bash
python predict_aqi.py predict --pm25 50 --no2 30 --co 1.5
```

### 2. Interactive Mode (Easiest)

```bash
python predict_aqi.py interactive
```

Follow the prompts to enter pollutant values.

### 3. Batch Prediction

```bash
python predict_aqi.py batch --file data.csv --output results.csv --show-stats
```

### 4. View Model Information

```bash
python predict_aqi.py model-info
```

### 5. Get Help

```bash
python predict_aqi.py --help
python predict_aqi.py predict --help
python predict_aqi.py batch --help
```

## Common Use Cases

### Quick AQI Check
```bash
python predict_aqi.py predict --pm25 75 --pm10 120 --no2 45
```

### Save Results to JSON
```bash
python predict_aqi.py predict --pm25 50 --no2 30 --json --output result.json
```

### Process Multiple Records
```bash
python predict_aqi.py batch --file measurements.csv --format json --output predictions.json
```

## Available Pollutants

- PM2.5, PM10 (particulate matter)
- NO, NO2, NOx (nitrogen compounds)
- NH3 (ammonia)
- CO (carbon monoxide)
- SO2 (sulfur dioxide)
- O3 (ozone)
- Benzene, Toluene, Xylene (volatile organic compounds)

You can provide any combination - missing values will be filled automatically.

