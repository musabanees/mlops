import numpy as np
import pandas as pd
import pickle
import yaml 
import json
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

def load_params( config_path='params.yaml' ):
    with open(config_path , "r") as file:
        params = yaml.safe_load(file)
    return params

def get_model(params):
    
    active_model = params.get('experiments', {}).get('Model' , 'gradient_boosting')

    print("="*50)
    print(f"🎯 DVC Selected Model: {active_model}")
    print("="*50)

    # Get model configuration
    model_config = params['Model'][active_model]
    model_type = model_config['type']
    model_params = model_config['params']

    print(f"Training {model_type} with params: {model_params}")

    if model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, active_model

def trained_model():

    params = load_params()
    model, model_name = get_model(params)

    # fetch the data from data/processed
    train_data = pd.read_csv('./data/features/train_bow.csv')

    X_train = train_data.iloc[:,0:-1].values
    y_train = train_data.iloc[:,-1].values

    # Define and train the model
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions , average='weighted' , zero_division=0)
    train_recall = recall_score(y_train, train_predictions, average='weighted' , zero_division=0)
    train_f1 = f1_score(y_train, train_predictions, average='weighted' , zero_division=0)

    os.makedirs('reports', exist_ok=True)
    metrics = {
        'model': model_name,
        'train_accuracy': float(train_accuracy),
        'train_precision': float(train_precision),
        'train_recall': float(train_recall),
        'train_f1': float(train_f1),
        'n_estimators': params['Model'][model_name]['params'].get('n_estimators'),
        'max_depth': params['Model'][model_name]['params'].get('max_depth')
    }

    with open('reports/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # save
    model_path = os.path.join('models', f'{model_name}.pkl')
    os.makedirs('models', exist_ok=True)
    pickle.dump(model, open(model_path,'wb'))
    print(f"Model saved to {model_path}")
    
    return model, metrics

if __name__ == '__main__':
    trained_model()