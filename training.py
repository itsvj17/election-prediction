import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

# Create a directory for saved models if it doesn't exist
model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load dataset
df = pd.read_csv("LS_2.0.csv")
df.columns = [col.replace('\n', ' ').replace('\r', ' ').strip() for col in df.columns]

print(df.shape)
print(df.describe())
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes) 

# Set style for visualizations
sns.set(style="whitegrid")
plt.figure(figsize=(18, 24))

# Loop through each column and plot
for i, column in enumerate(df.columns, 1):
    plt.subplot((len(df.columns) + 2) // 3, 3, i)
    if df[column].dtype == 'object':
        df[column].value_counts().head(10).plot(kind='bar', color='skyblue')
        plt.ylabel("Count")
    else:
        sns.histplot(df[column], bins=30, kde=True, color='salmon')
        plt.ylabel("Frequency")
    plt.title(column)
    plt.tight_layout()

plt.suptitle("Feature Distributions", fontsize=18, y=1.02)
plt.savefig(f"{model_dir}/feature_distributions.png")
plt.close()

# Clean currency fields
def clean_currency(value):
    if isinstance(value, str):
        value = value.split('~')[0].replace('Rs', '').replace(',', '').replace('+', '').strip()
        try:
            return float(value)
        except:
            return np.nan
    return value

df['CRIMINAL CASES'] = pd.to_numeric(df['CRIMINAL CASES'], errors='coerce')
df['ASSETS'] = df['ASSETS'].apply(clean_currency)
df['LIABILITIES'] = df['LIABILITIES'].apply(clean_currency)
print(df.isnull().sum())

# Drop rows with missing values in important columns
important_cols = ['PARTY', 'STATE', 'CONSTITUENCY', 'AGE', 'EDUCATION', 'GENDER',
                  'CRIMINAL CASES', 'ASSETS', 'LIABILITIES', 'TOTAL VOTES']
df = df.dropna(subset=important_cols).copy()

print(df.isnull().sum())

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ['PARTY', 'STATE', 'CONSTITUENCY', 'EDUCATION', 'GENDER']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders for future inference
with open(f"{model_dir}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Create binary WINNER target variable
df['WINNER'] = df.groupby('CONSTITUENCY')['TOTAL VOTES'].transform(lambda x: (x == x.max()).astype(int))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Use only numeric features
X = df.select_dtypes(include=[np.number]).drop(columns=['WINNER'])
y = df['WINNER']

# Save feature column names for inference
feature_columns = X.columns.tolist()
with open(f"{model_dir}/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future inference
with open(f"{model_dir}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train basic Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# Save basic Random Forest model
with open(f"{model_dir}/random_forest_basic.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Basic Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Basic Random Forest")
plt.tight_layout()
plt.savefig(f"{model_dir}/confusion_matrix_rf_basic.png")
plt.close()

# PCA
from sklearn.decomposition import PCA

# Apply PCA on scaled training data for analysis
pca_analysis = PCA()
pca_analysis.fit(X_train_scaled)

explained_var = pca_analysis.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--', label='Individual Explained Variance')
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='s', linestyle='-', label='Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - PCA Components')
plt.xticks(range(1, len(explained_var) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{model_dir}/pca_scree_plot.png")
plt.close()

print(f"Shape of Train Data before PCA: {X_train_scaled.shape}")
print(f"Shape of Test Data before PCA: {X_test_scaled.shape}")

# Apply PCA to retain 99% of variance
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save PCA transformer for future inference
with open(f"{model_dir}/pca_transformer.pkl", "wb") as f:
    pickle.dump(pca, f)

print(f"Shape of Train Data after reduction: {X_train_pca.shape}")
print(f"Shape of Test Data after reduction: {X_test_pca.shape}")

# Train Random Forest on PCA data
clf_pca = RandomForestClassifier(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)

# Save PCA Random Forest model
with open(f"{model_dir}/random_forest_pca.pkl", "wb") as f:
    pickle.dump(clf_pca, f)

print("\nRandom Forest with PCA:")
print("Accuracy after PCA:", accuracy_score(y_test, y_pred_pca))
print(classification_report(y_test, y_pred_pca))

cm_pca = confusion_matrix(y_test, y_pred_pca)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest with PCA")
plt.tight_layout()
plt.savefig(f"{model_dir}/confusion_matrix_rf_pca.png")
plt.close()

# Tuned models and stacking ensemble
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Random Forest with hyperparameter tuning
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_samples': [0.7, 0.9, None]
}

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit tuned RF
print("\nTraining tuned Random Forest model...")
rf_search.fit(X_train_pca, y_train)
rf_best = rf_search.best_estimator_

# Save tuned Random Forest model and its best parameters
with open(f"{model_dir}/random_forest_tuned.pkl", "wb") as f:
    pickle.dump(rf_best, f)

with open(f"{model_dir}/rf_best_params.pkl", "wb") as f:
    pickle.dump(rf_search.best_params_, f)

print(f"Best Random Forest parameters: {rf_search.best_params_}")

# Other base models
print("\nTraining XGBoost model...")
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train_pca, y_train)

# Save XGBoost model
with open(f"{model_dir}/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("\nTraining Gradient Boosting model...")
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_pca, y_train)

# Save Gradient Boosting model
with open(f"{model_dir}/gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(gb, f)

# Final stacking ensemble
print("\nTraining stacking ensemble...")
ensemble = StackingClassifier(
    estimators=[
        ('rf', rf_best),
        ('xgb', xgb),
        ('gb', gb)
    ],
    final_estimator=LogisticRegression(),
    passthrough=True,
    n_jobs=-1
)

# Train ensemble
ensemble.fit(X_train_pca, y_train)

# Save ensemble model
with open(f"{model_dir}/stacking_ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

y_pred_ensemble = ensemble.predict(X_test_pca)

# Evaluation
print("\nStacking Ensemble Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print(classification_report(y_test, y_pred_ensemble))

cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Stacking Ensemble")
plt.tight_layout()
plt.savefig(f"{model_dir}/confusion_matrix_ensemble.png")
plt.close()

# Create simple model metadata file
model_metadata = {
    "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "models": {
        "random_forest_basic": {
            "file": "random_forest_basic.pkl",
            "accuracy": float(accuracy_score(y_test, y_pred))
        },
        "random_forest_pca": {
            "file": "random_forest_pca.pkl",
            "accuracy": float(accuracy_score(y_test, y_pred_pca))
        },
        "random_forest_tuned": {
            "file": "random_forest_tuned.pkl",
            "parameters": rf_search.best_params_
        },
        "xgboost": {
            "file": "xgboost_model.pkl"
        },
        "gradient_boosting": {
            "file": "gradient_boosting_model.pkl"
        },
        "stacking_ensemble": {
            "file": "stacking_ensemble.pkl",
            "accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
            "base_models": ["random_forest_tuned", "xgboost", "gradient_boosting"]
        }
    },
    "preprocessing": {
        "scaler": "scaler.pkl",
        "pca": "pca_transformer.pkl",
        "label_encoders": "label_encoders.pkl",
        "feature_columns": "feature_columns.pkl"
    }
}

# Save metadata as JSON
import json
with open(f"{model_dir}/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=4)

print("\nAll models have been saved to the", model_dir, "directory")

# Create a simple prediction function demonstration
def make_prediction(model_path, new_data_csv, output_csv=None):
    """
    Make predictions using the saved model pipeline
    
    Parameters:
    - model_path: Path to the saved_models directory
    - new_data_csv: Path to CSV file with new data to predict
    - output_csv: Optional path to save predictions
    
    Returns:
    - DataFrame with predictions
    """
    # Load preprocessing components
    with open(f"{model_path}/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
        
    with open(f"{model_path}/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
        
    with open(f"{model_path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    with open(f"{model_path}/pca_transformer.pkl", "rb") as f:
        pca = pickle.load(f)
        
    # Load the ensemble model (best performer)
    with open(f"{model_path}/stacking_ensemble.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load and preprocess new data
    new_df = pd.read_csv(new_data_csv)
    
    # Clean the data similar to training
    new_df.columns = [col.replace('\n', ' ').replace('\r', ' ').strip() for col in new_df.columns]
    new_df['CRIMINAL CASES'] = pd.to_numeric(new_df['CRIMINAL CASES'], errors='coerce')
    new_df['ASSETS'] = new_df['ASSETS'].apply(clean_currency)
    new_df['LIABILITIES'] = new_df['LIABILITIES'].apply(clean_currency)
    
    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in new_df.columns:
            # Handle unseen categories by setting them to most frequent in training
            new_df[col] = new_df[col].map(
                lambda x: -1 if x not in encoder.classes_ else encoder.transform([x])[0]
            )
            # Replace -1 with most frequent class
            if (new_df[col] == -1).any():
                most_frequent_class = encoder.transform([encoder.classes_[0]])[0]
                new_df.loc[new_df[col] == -1, col] = most_frequent_class
    
    # Select numeric features
    X_new = new_df[feature_columns].copy()
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Apply PCA transformation
    X_new_pca = pca.transform(X_new_scaled)
    
    # Make predictions
    predictions = model.predict(X_new_pca)
    probabilities = model.predict_proba(X_new_pca)[:, 1]
    
    # Add predictions to the dataframe
    new_df['PREDICTED_WINNER'] = predictions
    new_df['WIN_PROBABILITY'] = probabilities
    
    # Save to CSV if specified
    if output_csv:
        new_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    
    return new_df

# Print example usage
print("\nExample usage of prediction function:")
print("from prediction_util import make_prediction")
print("predictions = make_prediction('saved_models', 'new_candidates.csv', 'predictions.csv')")