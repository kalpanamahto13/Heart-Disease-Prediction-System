import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

def main():
    try:
        # Load data
        data_path = 'data/heart.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure heart.csv is in data/ directory.")
        
        df = pd.read_csv(data_path)
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('?', np.nan)
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Load models and scaler
        models = {}
        model_names = ['logistic_regression', 'random_forest', 'svm']
        for name in model_names:
            model_path = f'models/{name}_model.pkl'
            if os.path.exists(model_path):
                models[name.replace('_', ' ').title()] = joblib.load(model_path)
        
        scaler = joblib.load('models/scaler.pkl')
        
        # Scale data
        X_scaled = scaler.transform(X)
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_scaled)
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrices.png')
        plt.close()
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig('visualizations/roc_curves.png')
        plt.close()
        
        # Feature importance for Random Forest
        if 'Random Forest' in models:
            rf_model = models['Random Forest']
            feature_names = X.columns
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Random Forest Feature Importance')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png')
            plt.close()
        
        # Model performance comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        # Assuming we have results from training, but since we don't, we'll compute on full data (not ideal but for demo)
        performance = {}
        for name, model in models.items():
            y_pred = model.predict(X_scaled)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            performance[name] = [
                accuracy_score(y, y_pred),
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                f1_score(y, y_pred)
            ]
        
        df_perf = pd.DataFrame(performance, index=metrics)
        df_perf.plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png')
        plt.close()
        
        print("Evaluation completed. Plots saved to visualizations/ directory.")
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")

if __name__ == "__main__":
    main()
