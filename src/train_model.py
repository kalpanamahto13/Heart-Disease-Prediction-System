import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def main():
    try:
        # Load data
        data_path = 'data/heart.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure heart.csv is in data/ directory.")
        
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")
        
        # Handle missing values - replace ? or NaN with median/mode
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('?', np.nan)
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
            
            if acc > best_score:
                best_score = acc
                best_model = model
        
        # Print results
        print("\nModel Performance:")
        for name, metrics in results.items():
            print(f"{name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Save best model as heart_model.pkl
        joblib.dump(best_model, 'models/heart_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        print(f"\nBest model ({best_model.__class__.__name__}) saved as models/heart_model.pkl")
        print("Scaler saved to models/scaler.pkl")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")

if __name__ == "__main__":
    main()
