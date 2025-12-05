#!/usr/bin/env python3
"""
CLI application for predicting AQI based on pollutant values
"""
import click
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

def load_model(model_path='models/aqi_model.pkl'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run 'python train_model.py' first."
        )
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    median_values = model_data.get('median_values', {})
    
    return model, features, median_values

def categorize_aqi(aqi_value):
    """Categorize AQI value into bucket"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"

def get_aqi_color(aqi_value):
    """Get color code for AQI value"""
    if aqi_value <= 50:
        return "green"
    elif aqi_value <= 100:
        return "yellow"
    elif aqi_value <= 200:
        return "bright_yellow"
    elif aqi_value <= 300:
        return "red"
    elif aqi_value <= 400:
        return "magenta"
    else:
        return "bright_red"

def get_health_recommendation(category):
    """Get health recommendations based on AQI category"""
    recommendations = {
        "Good": "Air quality is satisfactory. No health impacts expected.",
        "Satisfactory": "Sensitive people may experience minor breathing discomfort.",
        "Moderate": "People with lung disease, children, and older adults should reduce prolonged outdoor exertion.",
        "Poor": "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activities.",
        "Very Poor": "Health alert: Everyone may experience more serious health effects. Avoid outdoor activities.",
        "Severe": "Health warning: Emergency conditions. Entire population is likely to be affected."
    }
    return recommendations.get(category, "Unknown category")

def make_prediction(input_data, model_path='models/aqi_model.pkl', output_json=False, output_file=None):
    """Core prediction function that can be called from different commands"""
    model_obj, features, median_values = load_model(model_path)
    
    feature_map = {
        'pm25': 'PM2.5', 'pm2.5': 'PM2.5', 'PM2.5': 'PM2.5',
        'pm10': 'PM10', 'PM10': 'PM10',
        'no': 'NO', 'NO': 'NO',
        'no2': 'NO2', 'NO2': 'NO2',
        'nox': 'NOx', 'NOx': 'NOx',
        'nh3': 'NH3', 'NH3': 'NH3',
        'co': 'CO', 'CO': 'CO',
        'so2': 'SO2', 'SO2': 'SO2',
        'o3': 'O3', 'O3': 'O3',
        'benzene': 'Benzene', 'Benzene': 'Benzene',
        'toluene': 'Toluene', 'Toluene': 'Toluene',
        'xylene': 'Xylene', 'Xylene': 'Xylene'
    }
    
    normalized_input = {}
    provided_pollutants = {}
    for key, value in input_data.items():
        if value is not None:
            feature_name = feature_map.get(key.lower(), key)
            normalized_input[feature_name] = value
            provided_pollutants[feature_name] = value
    
    input_df = pd.DataFrame([{f: None for f in features}])
    
    for feature in features:
        if feature in normalized_input:
            input_df[feature] = normalized_input[feature]
    
    imputed_values = {}
    for feature in features:
        if pd.isna(input_df[feature].iloc[0]):
            imputed_val = median_values.get(feature, 0.0)
            input_df.loc[:, feature] = imputed_val
            imputed_values[feature] = imputed_val
    
    prediction = model_obj.predict(input_df[features])[0]
    category = categorize_aqi(prediction)
    recommendation = get_health_recommendation(category)
    
    return {
        'prediction': prediction,
        'category': category,
        'recommendation': recommendation,
        'provided_pollutants': provided_pollutants,
        'imputed_values': imputed_values
    }

def validate_pollutant_value(name, value):
    """Validate pollutant value is non-negative"""
    if value is not None and value < 0:
        raise click.BadParameter(f"{name} cannot be negative. Got {value}")
    return value

@click.group()
def cli():
    """AQI Prediction CLI - Predict Air Quality Index from pollutant values"""
    pass

@cli.command()
@click.option('--pm25', type=float, callback=lambda ctx, param, value: validate_pollutant_value('PM2.5', value), help='PM2.5 concentration (μg/m³)')
@click.option('--pm10', type=float, callback=lambda ctx, param, value: validate_pollutant_value('PM10', value), help='PM10 concentration (μg/m³)')
@click.option('--no', type=float, callback=lambda ctx, param, value: validate_pollutant_value('NO', value), help='NO concentration (μg/m³)')
@click.option('--no2', type=float, callback=lambda ctx, param, value: validate_pollutant_value('NO2', value), help='NO2 concentration (μg/m³)')
@click.option('--nox', type=float, callback=lambda ctx, param, value: validate_pollutant_value('NOx', value), help='NOx concentration (μg/m³)')
@click.option('--nh3', type=float, callback=lambda ctx, param, value: validate_pollutant_value('NH3', value), help='NH3 concentration (μg/m³)')
@click.option('--co', type=float, callback=lambda ctx, param, value: validate_pollutant_value('CO', value), help='CO concentration (mg/m³)')
@click.option('--so2', type=float, callback=lambda ctx, param, value: validate_pollutant_value('SO2', value), help='SO2 concentration (μg/m³)')
@click.option('--o3', type=float, callback=lambda ctx, param, value: validate_pollutant_value('O3', value), help='O3 concentration (μg/m³)')
@click.option('--benzene', type=float, callback=lambda ctx, param, value: validate_pollutant_value('Benzene', value), help='Benzene concentration (μg/m³)')
@click.option('--toluene', type=float, callback=lambda ctx, param, value: validate_pollutant_value('Toluene', value), help='Toluene concentration (μg/m³)')
@click.option('--xylene', type=float, callback=lambda ctx, param, value: validate_pollutant_value('Xylene', value), help='Xylene concentration (μg/m³)')
@click.option('--model', default='models/aqi_model.pkl', help='Path to trained model')
@click.option('--json', 'output_json', is_flag=True, help='Output results in JSON format')
@click.option('--output', type=click.Path(), help='Save results to file (JSON or CSV)')
def predict(pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, model, output_json, output):
    """Predict AQI from pollutant values"""
    try:
        # Create input dictionary
        input_data = {
            'pm25': pm25,
            'pm10': pm10,
            'no': no,
            'no2': no2,
            'nox': nox,
            'nh3': nh3,
            'co': co,
            'so2': so2,
            'o3': o3,
            'benzene': benzene,
            'toluene': toluene,
            'xylene': xylene
        }
        
        provided_pollutants = {k: v for k, v in input_data.items() if v is not None}
        if not provided_pollutants:
            click.echo("Error: Please provide at least one pollutant value.", err=True)
            click.echo("\nExample usage:")
            click.echo("  python predict_aqi.py predict --pm25 50 --no2 30 --co 1.5")
            click.echo("\nOr use interactive mode:")
            click.echo("  python predict_aqi.py interactive")
            return
        
        result = make_prediction(input_data, model)
        prediction = result['prediction']
        category = result['category']
        recommendation = result['recommendation']
        provided = result['provided_pollutants']
        imputed = result['imputed_values']
        
        results = {
            'predicted_aqi': round(prediction, 2),
            'aqi_category': category,
            'health_recommendation': recommendation,
            'input_pollutants': {k: v for k, v in provided.items()},
            'imputed_pollutants': {k: v for k, v in imputed.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        if output_json or (output and output.endswith('.json')):
            json_output = json.dumps(results, indent=2)
            if output:
                with open(output, 'w') as f:
                    f.write(json_output)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(json_output)
            return
        
        color = get_aqi_color(prediction)
        click.echo("\n" + "=" * 70)
        click.echo(click.style("AQI Prediction Results", bold=True, fg='cyan'))
        click.echo("=" * 70)
        click.echo(f"\n{click.style('Predicted AQI:', bold=True)} {click.style(f'{prediction:.2f}', bold=True, fg=color)}")
        click.echo(f"{click.style('AQI Category:', bold=True)} {click.style(category, fg=color, bold=True)}")
        click.echo(f"\n{click.style('Health Recommendation:', bold=True)}")
        click.echo(f"  {recommendation}")
        
        click.echo(f"\n{click.style('Input Pollutants:', bold=True)}")
        for pollutant, value in provided.items():
            unit = "μg/m³" if pollutant not in ['CO'] else "mg/m³"
            click.echo(f"  • {pollutant}: {value} {unit}")
        
        if imputed:
            click.echo(f"\n{click.style('Imputed Pollutants (using median values):', bold=True, fg='yellow')}")
            for pollutant, value in imputed.items():
                unit = "μg/m³" if pollutant not in ['CO'] else "mg/m³"
                click.echo(f"  • {pollutant}: {value:.2f} {unit}")
        
        if output:
            if output.endswith('.csv'):
                df = pd.DataFrame([{
                    'Predicted_AQI': prediction,
                    'AQI_Category': category,
                    **provided
                }])
                df.to_csv(output, index=False)
                click.echo(f"\n{click.style('Results saved to:', bold=True)} {output}")
            else:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
                click.echo(f"\n{click.style('Results saved to:', bold=True)} {output}")
        
        click.echo("\n" + "=" * 70)
        
    except FileNotFoundError as e:
        click.echo(click.style(f"Error: {e}", fg='red', bold=True), err=True)
    except click.BadParameter as e:
        click.echo(click.style(f"Error: {e}", fg='red', bold=True), err=True)
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red', bold=True), err=True)

@cli.command()
@click.option('--file', type=click.Path(exists=True), required=True, help='CSV file with pollutant data')
@click.option('--model', default='models/aqi_model.pkl', help='Path to trained model')
@click.option('--output', default='predictions.csv', help='Output file for predictions')
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv', help='Output format')
@click.option('--show-stats', is_flag=True, help='Show statistics about predictions')
def batch(file, model, output, format, show_stats):
    """Predict AQI for multiple records from a CSV file"""
    try:
        click.echo(click.style("\nLoading model and data...", fg='cyan'))
        
        model_obj, features, median_values = load_model(model)
        df = pd.read_csv(file)
        click.echo(f"Loaded {len(df)} records from {file}")
        
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            click.echo(click.style(f"Warning: Missing columns {missing_cols}. Will use median values.", fg='yellow'))
        
        X = pd.DataFrame()
        for feature in features:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                X[feature] = median_values.get(feature, 0.0)
        
        for feature in features:
            X[feature] = X[feature].fillna(median_values.get(feature, 0.0))
        
        click.echo("Making predictions...")
        predictions = model_obj.predict(X[features])
        categories = [categorize_aqi(pred) for pred in predictions]
        recommendations = [get_health_recommendation(cat) for cat in categories]
        
        result_df = df.copy()
        result_df['Predicted_AQI'] = predictions.round(2)
        result_df['Predicted_AQI_Bucket'] = categories
        result_df['Health_Recommendation'] = recommendations
        
        if format == 'json':
            output = output.replace('.csv', '.json') if output.endswith('.csv') else output
            results_list = []
            for idx, row in result_df.iterrows():
                results_list.append({
                    'record_id': idx,
                    'predicted_aqi': float(row['Predicted_AQI']),
                    'aqi_category': row['Predicted_AQI_Bucket'],
                    'health_recommendation': row['Health_Recommendation'],
                    'pollutants': {col: float(row[col]) if pd.notna(row[col]) else None 
                                 for col in features if col in row.index}
                })
            with open(output, 'w') as f:
                json.dump(results_list, f, indent=2)
        else:
            result_df.to_csv(output, index=False)
        
        click.echo(click.style(f"\n✓ Predictions completed!", fg='green', bold=True))
        click.echo(f"Total records: {len(df)}")
        click.echo(f"Results saved to: {output}")
        
        # Show statistics
        if show_stats:
            click.echo("\n" + "=" * 70)
            click.echo(click.style("Prediction Statistics", bold=True, fg='cyan'))
            click.echo("=" * 70)
            click.echo(f"\nAQI Statistics:")
            click.echo(f"  Mean: {predictions.mean():.2f}")
            click.echo(f"  Median: {np.median(predictions):.2f}")
            click.echo(f"  Min: {predictions.min():.2f}")
            click.echo(f"  Max: {predictions.max():.2f}")
            click.echo(f"  Std Dev: {predictions.std():.2f}")
            
            click.echo(f"\nCategory Distribution:")
            category_counts = pd.Series(categories).value_counts().sort_index()
            for cat, count in category_counts.items():
                percentage = (count / len(categories)) * 100
                color = get_aqi_color({'Good': 25, 'Satisfactory': 75, 'Moderate': 150, 
                                      'Poor': 250, 'Very Poor': 350, 'Severe': 450}.get(cat, 0))
                click.echo(f"  {click.style(cat, fg=color)}: {count} ({percentage:.1f}%)")
        
        click.echo(f"\n{click.style('Sample predictions:', bold=True)}")
        sample_cols = ['Predicted_AQI', 'Predicted_AQI_Bucket']
        if len(result_df.columns) <= 15:  # Show all columns if not too many
            click.echo(result_df[sample_cols + [c for c in features if c in result_df.columns]].head(10).to_string(index=False))
        else:
            click.echo(result_df[sample_cols].head(10).to_string(index=False))
        
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red', bold=True), err=True)

@cli.command()
def interactive():
    """Interactive mode for entering pollutant values"""
    click.echo(click.style("\n" + "=" * 70, fg='cyan'))
    click.echo(click.style("Interactive AQI Prediction", bold=True, fg='cyan'))
    click.echo("=" * 70)
    click.echo("\nEnter pollutant values (press Enter to skip):\n")
    
    pollutants = {
        'PM2.5': {'unit': 'μg/m³', 'desc': 'Particulate Matter 2.5'},
        'PM10': {'unit': 'μg/m³', 'desc': 'Particulate Matter 10'},
        'NO': {'unit': 'μg/m³', 'desc': 'Nitrogen Oxide'},
        'NO2': {'unit': 'μg/m³', 'desc': 'Nitrogen Dioxide'},
        'NOx': {'unit': 'μg/m³', 'desc': 'Nitrogen Oxides'},
        'NH3': {'unit': 'μg/m³', 'desc': 'Ammonia'},
        'CO': {'unit': 'mg/m³', 'desc': 'Carbon Monoxide'},
        'SO2': {'unit': 'μg/m³', 'desc': 'Sulfur Dioxide'},
        'O3': {'unit': 'μg/m³', 'desc': 'Ozone'},
        'Benzene': {'unit': 'μg/m³', 'desc': 'Benzene'},
        'Toluene': {'unit': 'μg/m³', 'desc': 'Toluene'},
        'Xylene': {'unit': 'μg/m³', 'desc': 'Xylene'}
    }
    
    input_data = {}
    for pollutant, info in pollutants.items():
        while True:
            try:
                value = click.prompt(f"{pollutant} ({info['unit']})", default='', show_default=False)
                if value == '':
                    break
                value = float(value)
                if value < 0:
                    click.echo(click.style("  Error: Value cannot be negative. Please try again.", fg='red'))
                    continue
                input_data[pollutant.lower().replace('.', '')] = value
                break
            except ValueError:
                click.echo(click.style("  Error: Invalid number. Please try again.", fg='red'))
            except KeyboardInterrupt:
                click.echo("\n\nCancelled.")
                return
    
    if not input_data:
        click.echo(click.style("\nError: No pollutant values provided.", fg='red'))
        return
    
    # Make prediction using the shared function
    try:
        result = make_prediction(input_data)
        prediction = result['prediction']
        category = result['category']
        recommendation = result['recommendation']
        provided = result['provided_pollutants']
        imputed = result['imputed_values']
        
        # Display results
        color = get_aqi_color(prediction)
        click.echo("\n" + "=" * 70)
        click.echo(click.style("AQI Prediction Results", bold=True, fg='cyan'))
        click.echo("=" * 70)
        click.echo(f"\n{click.style('Predicted AQI:', bold=True)} {click.style(f'{prediction:.2f}', bold=True, fg=color)}")
        click.echo(f"{click.style('AQI Category:', bold=True)} {click.style(category, fg=color, bold=True)}")
        click.echo(f"\n{click.style('Health Recommendation:', bold=True)}")
        click.echo(f"  {recommendation}")
        
        click.echo(f"\n{click.style('Input Pollutants:', bold=True)}")
        for pollutant, value in provided.items():
            unit = "μg/m³" if pollutant not in ['CO'] else "mg/m³"
            click.echo(f"  • {pollutant}: {value} {unit}")
        
        if imputed:
            click.echo(f"\n{click.style('Imputed Pollutants (using median values):', bold=True, fg='yellow')}")
            for pollutant, value in imputed.items():
                unit = "μg/m³" if pollutant not in ['CO'] else "mg/m³"
                click.echo(f"  • {pollutant}: {value:.2f} {unit}")
        
        click.echo("\n" + "=" * 70)
        
    except FileNotFoundError as e:
        click.echo(click.style(f"Error: {e}", fg='red', bold=True), err=True)
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red', bold=True), err=True)

@cli.command()
@click.option('--model', default='models/aqi_model.pkl', help='Path to trained model')
def model_info(model):
    """Display information about the trained model"""
    try:
        model_obj, features, median_values = load_model(model)
        
        click.echo("\n" + "=" * 70)
        click.echo(click.style("Model Information", bold=True, fg='cyan'))
        click.echo("=" * 70)
        
        click.echo(f"\n{click.style('Model Type:', bold=True)} Random Forest Regressor")
        click.echo(f"{click.style('Number of Features:', bold=True)} {len(features)}")
        click.echo(f"{click.style('Number of Estimators:', bold=True)} {model_obj.n_estimators}")
        click.echo(f"{click.style('Max Depth:', bold=True)} {model_obj.max_depth}")
        
        click.echo(f"\n{click.style('Features:', bold=True)}")
        for i, feature in enumerate(features, 1):
            click.echo(f"  {i:2d}. {feature}")
        
        click.echo(f"\n{click.style('Feature Importance (Top 10):', bold=True)}")
        importances = pd.DataFrame({
            'feature': features,
            'importance': model_obj.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(10).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            click.echo(f"  {row['feature']:12s} {bar} {row['importance']:.4f}")
        
        click.echo(f"\n{click.style('Median Values (for imputation):', bold=True)}")
        for feature in features:
            unit = "μg/m³" if feature != 'CO' else "mg/m³"
            click.echo(f"  {feature:12s}: {median_values.get(feature, 0.0):.2f} {unit}")
        
        click.echo("\n" + "=" * 70)
        
    except FileNotFoundError as e:
        click.echo(click.style(f"Error: {e}", fg='red', bold=True), err=True)
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg='red', bold=True), err=True)

@cli.command()
def info():
    """Display information about the AQI prediction system"""
    click.echo("\n" + "=" * 70)
    click.echo(click.style("AQI Prediction System Information", bold=True, fg='cyan'))
    click.echo("=" * 70)
    click.echo("\n" + click.style("Supported Pollutants:", bold=True))
    pollutants_info = [
        ("PM2.5", "μg/m³", "Particulate Matter 2.5"),
        ("PM10", "μg/m³", "Particulate Matter 10"),
        ("NO", "μg/m³", "Nitrogen Oxide"),
        ("NO2", "μg/m³", "Nitrogen Dioxide"),
        ("NOx", "μg/m³", "Nitrogen Oxides"),
        ("NH3", "μg/m³", "Ammonia"),
        ("CO", "mg/m³", "Carbon Monoxide"),
        ("SO2", "μg/m³", "Sulfur Dioxide"),
        ("O3", "μg/m³", "Ozone"),
        ("Benzene", "μg/m³", "Benzene"),
        ("Toluene", "μg/m³", "Toluene"),
        ("Xylene", "μg/m³", "Xylene")
    ]
    for name, unit, desc in pollutants_info:
        click.echo(f"  • {name:10s} ({unit:8s}) - {desc}")
    
    click.echo("\n" + click.style("AQI Categories:", bold=True))
    categories = [
        (0, 50, "Good", "green", "Minimal impact"),
        (51, 100, "Satisfactory", "yellow", "Minor breathing discomfort"),
        (101, 200, "Moderate", "bright_yellow", "Breathing discomfort to people with lung disease"),
        (201, 300, "Poor", "red", "Breathing discomfort to most people"),
        (301, 400, "Very Poor", "magenta", "Respiratory illness on prolonged exposure"),
        (401, None, "Severe", "bright_red", "Health hazard")
    ]
    for min_val, max_val, name, color, desc in categories:
        range_str = f"{min_val}-{max_val}" if max_val else f"{min_val}+"
        click.echo(f"  {click.style(name, fg=color, bold=True):15s} ({range_str:6s}): {desc}")
    
    click.echo("\n" + click.style("Available Commands:", bold=True))
    click.echo("  predict      - Predict AQI from command-line arguments")
    click.echo("  interactive  - Interactive mode for entering pollutant values")
    click.echo("  batch        - Batch prediction from CSV file")
    click.echo("  model-info   - Display information about the trained model")
    click.echo("  info         - Display this help information")
    
    click.echo("\n" + click.style("Example Usage:", bold=True))
    click.echo("  # Single prediction:")
    click.echo("  python predict_aqi.py predict --pm25 50 --no2 30 --co 1.5")
    click.echo("\n  # Interactive mode:")
    click.echo("  python predict_aqi.py interactive")
    click.echo("\n  # Batch prediction:")
    click.echo("  python predict_aqi.py batch --file data.csv --output results.csv --show-stats")
    click.echo("\n  # Model information:")
    click.echo("  python predict_aqi.py model-info")
    click.echo("\n" + "=" * 70)

if __name__ == '__main__':
    cli()

