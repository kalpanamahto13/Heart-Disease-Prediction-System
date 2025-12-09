import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Handle missing values for EDA
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('?', np.nan)
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Outcome distribution
        plt.figure(figsize=(6,4))
        sns.countplot(x='target', data=df)
        plt.title('Heart Disease Distribution')
        plt.xlabel('Target (0=No Disease, 1=Disease)')
        plt.ylabel('Count')
        plt.savefig('visualizations/outcome_distribution.png')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12,10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.savefig('visualizations/correlation_heatmap.png')
        plt.close()
        
        # Feature-wise distributions
        features = df.columns.drop('target')
        for feature in features:
            plt.figure(figsize=(6,4))
            if df[feature].dtype == 'object' or df[feature].nunique() < 10:
                sns.countplot(x=feature, hue='target', data=df)
                plt.title(f'{feature} Distribution by Target')
            else:
                sns.histplot(data=df, x=feature, hue='target', kde=True)
                plt.title(f'{feature} Distribution by Target')
            plt.savefig(f'visualizations/{feature}_distribution.png')
            plt.close()
        
        print("EDA completed. Visualizations saved to visualizations/ directory.")
        
    except Exception as e:
        print(f"Error in EDA: {str(e)}")

if __name__ == "__main__":
    main()
