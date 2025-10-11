import numpy as np
import pandas as pd
import pickle
import json
import os
import yaml
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, f1_score, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

def load_params(config_path='params.yaml'):
    """Load parameters from params.yaml"""
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)
    return params


def evaluate_model():
    """Evaluate the trained model on test data"""
    params = load_params()
    
    # Determine which model was used for training
    active_model = params.get('experiments', {}).get('Model', 'gradient_boosting')
    print(f"Evaluating model: {active_model}")
    
    # Load the model
    model_path = os.path.join('models', f'{active_model}.pkl')
    
    # Check if model exists, otherwise use the default model.pkl
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Using model.pkl instead.")
        model_path = 'model.pkl'
        
    try:
        model = pickle.load(open(model_path, 'rb'))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Load test data
    try:
        test_data = pd.read_csv('./data/features/test_bow.csv')
    except FileNotFoundError:
        print("Error: Test data file not found!")
        return
    
    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For ROC-AUC, we need probability predictions
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    else:
        has_proba = False
        
    # Calculate evaluation metrics
    metrics = {
        'model': active_model,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='binary', zero_division=0))
    }
    
    # Add AUC if probabilities are available
    if has_proba:
        metrics['auc'] = float(roc_auc_score(y_test, y_pred_proba))
    
    # Print evaluation results
    print("\n" + "="*50)
    print("Model Evaluation Results:")
    print(f"Model: {active_model}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if has_proba:
        print(f"AUC-ROC: {metrics['auc']:.4f}")
    print("="*50 + "\n")
    
    # Create directories for outputs if they don't exist
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Save metrics
    with open('reports/test_metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    
    # Also save as metrics.json for backward compatibility
    with open('reports/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    
    # Generate and save confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {active_model}')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{active_model}.png')
    
    # If probabilities are available, generate ROC curve
    if has_proba:
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {active_model}')
        plt.legend(loc='lower right')
        plt.savefig(f'reports/figures/roc_curve_{active_model}.png')
    
    return metrics

if __name__ == '__main__':
    evaluate_model()



