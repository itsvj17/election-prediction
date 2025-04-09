# app.py
import os
import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import io
import csv
from datetime import datetime

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'saved_models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_currency(value):
    if isinstance(value, str):
        value = value.split('~')[0].replace('Rs', '').replace(',', '').replace('+', '').strip()
        try:
            return float(value)
        except:
            return np.nan
    return value

def load_model_components():
    """Load all the necessary model components"""
    model_path = app.config['MODEL_FOLDER']
    components = {}
    
    # Load preprocessing components
    with open(f"{model_path}/label_encoders.pkl", "rb") as f:
        components['label_encoders'] = pickle.load(f)
        
    with open(f"{model_path}/feature_columns.pkl", "rb") as f:
        components['feature_columns'] = pickle.load(f)
        
    with open(f"{model_path}/scaler.pkl", "rb") as f:
        components['scaler'] = pickle.load(f)
        
    with open(f"{model_path}/pca_transformer.pkl", "rb") as f:
        components['pca'] = pickle.load(f)
    
    # Load model metadata to display info about available models
    try:
        with open(f"{model_path}/model_metadata.json", "r") as f:
            components['metadata'] = json.load(f)
    except:
        components['metadata'] = {"models": {"stacking_ensemble": {"file": "stacking_ensemble.pkl"}}}
    
    # Load default model (stacking ensemble)
    with open(f"{model_path}/stacking_ensemble.pkl", "rb") as f:
        components['model'] = pickle.load(f)
    
    return components

def predict_with_model(df, components, model_name='stacking_ensemble.pkl'):
    """
    Make predictions using the loaded model components
    
    Parameters:
    - df: DataFrame with candidate data
    - components: Dictionary of model components
    - model_name: Name of the model file to use
    
    Returns:
    - DataFrame with predictions added
    """
    # If a different model is requested, load it
    if model_name != 'stacking_ensemble.pkl' and 'model' in components:
        model_path = app.config['MODEL_FOLDER']
        with open(f"{model_path}/{model_name}", "rb") as f:
            components['model'] = pickle.load(f)
    
    # Ensure column names are clean
    df.columns = [col.replace('\n', ' ').replace('\r', ' ').strip() for col in df.columns]
    
    # Clean numeric fields
    df['CRIMINAL CASES'] = pd.to_numeric(df['CRIMINAL CASES'], errors='coerce')
    df['ASSETS'] = df['ASSETS'].apply(clean_currency)
    df['LIABILITIES'] = df['LIABILITIES'].apply(clean_currency)
    
    # Encode categorical features
    for col, encoder in components['label_encoders'].items():
        if col in df.columns:
            # Handle unseen categories
            df[col] = df[col].map(
                lambda x: -1 if x not in encoder.classes_ else encoder.transform([x])[0]
            )
            # Replace -1 with most frequent class
            if (df[col] == -1).any():
                most_frequent_class = encoder.transform([encoder.classes_[0]])[0]
                df.loc[df[col] == -1, col] = most_frequent_class
    
    # Select and arrange features in the correct order
    X_new = pd.DataFrame(index=df.index)
    for col in components['feature_columns']:
        if col in df.columns:
            X_new[col] = df[col]
        else:
            X_new[col] = 0  # Default value if column is missing
    
    # Scale the features
    X_new_scaled = components['scaler'].transform(X_new)
    
    # Apply PCA transformation
    X_new_pca = components['pca'].transform(X_new_scaled)
    
    # Make predictions
    predictions = components['model'].predict(X_new_pca)
    probabilities = components['model'].predict_proba(X_new_pca)[:, 1]
    
    # Add predictions to the dataframe
    df['PREDICTED_WINNER'] = predictions
    df['WIN_PROBABILITY'] = probabilities
    
    return df

def get_categorical_values():
    """Get allowed values for categorical fields from label encoders"""
    model_path = app.config['MODEL_FOLDER']
    with open(f"{model_path}/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    
    categorical_values = {}
    for col, encoder in label_encoders.items():
        categorical_values[col] = encoder.classes_.tolist()
    
    return categorical_values

# Load all model components on startup
model_components = load_model_components()
categorical_values = get_categorical_values()

# Routes
@app.route('/')
def index():
    """Landing page with options for batch prediction or single prediction"""
    return render_template('index.html', 
                          models=model_components['metadata']['models'])

@app.route('/predict-batch', methods=['GET', 'POST'])
def predict_batch():
    """Handle batch prediction through CSV file upload"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('predict_batch.html', error='No file part')
        
        file = request.files['file']
        model_name = request.form.get('model', 'stacking_ensemble.pkl')
        
        # If user does not select file, browser might submit an empty part
        if file.filename == '':
            return render_template('predict_batch.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            # Read the CSV file
            try:
                df = pd.read_csv(file)
                
                # Run predictions
                result_df = predict_with_model(df, model_components, model_name)
                
                # Convert the result to CSV for download
                output = io.StringIO()
                result_df.to_csv(output, index=False)
                
                # Create a response with the CSV file
                response = app.response_class(
                    response=output.getvalue(),
                    mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
                )
                
                return response
                
            except Exception as e:
                return render_template('predict_batch.html', 
                                      error=f'Error processing file: {str(e)}',
                                      models=model_components['metadata']['models'])
        else:
            return render_template('predict_batch.html', 
                                  error='File type not allowed. Please upload a CSV file.',
                                  models=model_components['metadata']['models'])
    
    # GET request - show the upload form
    return render_template('predict_batch.html', 
                          models=model_components['metadata']['models'])

@app.route('/predict-single', methods=['GET', 'POST'])
def predict_single():
    """Handle single prediction through a web form"""
    if request.method == 'POST':
        try:
            # Create a single-row DataFrame from form data
            form_data = {}
            
            # Get form data and convert to appropriate types
            for field in request.form:
                if field == 'model':
                    continue  # Skip model selection field
                
                value = request.form[field]
                if field in ['AGE', 'CRIMINAL CASES']:
                    form_data[field] = float(value) if value else np.nan
                elif field in ['ASSETS', 'LIABILITIES', 'TOTAL VOTES']:
                    form_data[field] = float(value.replace(',', '')) if value else np.nan
                else:
                    form_data[field] = value
            
            df = pd.DataFrame([form_data])
            model_name = request.form.get('model', 'stacking_ensemble.pkl')
            
            # Run prediction
            result_df = predict_with_model(df, model_components, model_name)
            
            # Extract prediction results
            prediction = {
                'predicted_winner': bool(result_df['PREDICTED_WINNER'].iloc[0]),
                'win_probability': float(result_df['WIN_PROBABILITY'].iloc[0]),
                'input_data': form_data
            }
            
            return render_template('predict_single.html', 
                                  prediction=prediction,
                                  categorical_values=categorical_values,
                                  models=model_components['metadata']['models'])
            
        except Exception as e:
            return render_template('predict_single.html', 
                                  error=f'Error making prediction: {str(e)}',
                                  categorical_values=categorical_values,
                                  models=model_components['metadata']['models'])
    
    # GET request - show the prediction form
    return render_template('predict_single.html', 
                          categorical_values=categorical_values,
                          models=model_components['metadata']['models'])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for batch prediction"""
    try:
        # Check if request has file or JSON data
        if 'file' in request.files:
            file = request.files['file']
            model_name = request.form.get('model', 'stacking_ensemble.pkl')
            
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
            else:
                return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
                
        elif request.is_json:
            # Handle JSON payload (list of records)
            data = request.json
            model_name = data.pop('model', 'stacking_ensemble.pkl')
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'candidates' in data:
                df = pd.DataFrame(data['candidates'])
            else:
                return jsonify({'error': 'Invalid JSON format. Expecting list of records or {candidates: [...]} format'}), 400
        else:
            return jsonify({'error': 'No data provided. Send either a CSV file or JSON records.'}), 400
        
        # Process data and make predictions
        result_df = predict_with_model(df, model_components, model_name)
        
        # Return predictions as JSON
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append({
                'predicted_winner': bool(row['PREDICTED_WINNER']),
                'win_probability': float(row['WIN_PROBABILITY']),
                'input_data': row.drop(['PREDICTED_WINNER', 'WIN_PROBABILITY']).to_dict()
            })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Display information about the trained models"""
    return render_template('model_info.html', 
                          metadata=model_components['metadata'],
                          feature_columns=model_components['feature_columns'])

# Add example data upload
@app.route('/download-template')
def download_template():
    """Provide a template CSV file for batch predictions"""
    # Create a minimal template with required columns
    required_columns = ['PARTY', 'STATE', 'CONSTITUENCY', 'AGE', 'EDUCATION', 
                        'GENDER', 'CRIMINAL CASES', 'ASSETS', 'LIABILITIES', 'TOTAL VOTES']
    
    # Create a CSV string with just the headers
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(required_columns)
    
    # Create a response with the CSV file
    response = app.response_class(
        response=output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=candidate_template.csv"}
    )
    
    return response

if __name__ == '__main__':
    app.run(debug=True)