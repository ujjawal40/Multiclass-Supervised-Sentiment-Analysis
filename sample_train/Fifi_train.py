# ------------------------Libraries------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, f1_score
from joblib import Parallel, delayed, dump
import pickle
import time
import os
import sys

nickname = 'Fifi'

# ------------------------Functions------------------------------------
def read_data(path):
    try:
        df = pd.read_excel(path)
        return df
    except FileNotFoundError:
        print(f"Error: File {path} not found. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading data: {e}")
        sys.exit(1)

def split(df, split='test'):
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == split]
    if train_df.empty or test_df.empty:
        print(f"Warning: No data found for split 'train' or '{split}'. Check your data.")
    return train_df, test_df

def oversample_dataframe(df, label_col='Label'):

    max_count = df[label_col].value_counts().max()
    df_list = []
    for label, group in df.groupby(label_col):
        group_upsampled = group.sample(max_count, replace=True, random_state=42)
        df_list.append(group_upsampled)
    return pd.concat(df_list)

def model_training(train_df, label_encoder, model_name, model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)),
        ('clf', model)
    ])

    # Encode labels
    train_df = train_df.copy()
    # Ensure label encoder is fitted before using it
    label_encoder.fit(train_df['Label'])
    train_df['Label_enc'] = label_encoder.transform(train_df['Label'])

    # Fit the pipeline
    start_time = time.time()
    pipeline.fit(train_df['Text'], train_df['Label_enc'])
    end_time = time.time()

    return pipeline, end_time - start_time

def model_testing(pipeline, test_df):
    if test_df.empty or 'Text' not in test_df.columns:
        print("Error: Test data is empty or missing 'Text' column.")
        return np.array([])
    predictions = pipeline.predict(test_df['Text'])
    return predictions

def metrics(predictions, test_df, label_encoder):
    if len(predictions) == 0 or test_df.empty or 'Label' not in test_df.columns:
        print("Error: Cannot compute metrics due to empty predictions or test labels.")
        return 0.0
    # Decode predictions if needed
    predictions = label_encoder.inverse_transform(predictions)
    test_labels = test_df['Label'].values
    kappa = cohen_kappa_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='macro')
    return 0.5 * kappa + 0.5 * f1

def save_model(model, file_name):
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Error saving model to {file_name}: {e}")
        sys.exit(1)

def save_vectorizer(vectorizer, file_name):
    try:
        dump(vectorizer, file_name)
    except Exception as e:
        print(f"Error saving vectorizer to {file_name}: {e}")
        sys.exit(1)

def save_label_encoder(label_encoder, file_name):
    try:
        dump(label_encoder, file_name)
    except Exception as e:
        print(f"Error saving label encoder to {file_name}: {e}")
        sys.exit(1)

# ------------------------Main Function------------------------------------
def train_and_evaluate(model_name, model, train_df, test_df, label_encoder):
    # Train the model
    pipeline, training_time = model_training(train_df, label_encoder, model_name, model)

    # Test the model
    predictions = model_testing(pipeline, test_df)

    # Compute metrics
    metric_score = metrics(predictions, test_df, label_encoder)

    return {
        'model_name': model_name,
        'pipeline': pipeline,
        'metric_score': metric_score,
        'training_time': training_time
    }

def main():
    # Load and split data
    df = read_data(path="../Data/train.xlsx")
    train_df, test_df = split(df, split='test')

    # Define label encoder
    label_encoder = LabelEncoder()

    # Define models
    models = {
        'SVM': OneVsRestClassifier(LinearSVC(random_state=42)),
        'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    # Train and evaluate models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate)(name, model, train_df, test_df, label_encoder)
        for name, model in models.items()
    )

    # Print results
    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"Metric Score (0.5*kappa + 0.5*f1): {result['metric_score']:.4f}")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        print("---")

    # Find the best model
    best_result = max(results, key=lambda x: x['metric_score'])
    print(f"Best Model: {best_result['model_name']} with Metric Score: {best_result['metric_score']:.4f}")

    # Save the best model, vectorizer, and label encoder
    best_pipeline = best_result['pipeline']
    save_model(best_pipeline, f'{nickname}_model.pickle')
    save_vectorizer(best_pipeline.named_steps['tfidf'], f'{nickname}_vectorizer.joblib')
    save_label_encoder(label_encoder, f'{nickname}_labelencoder.joblib')
    print(f"Saved best model ({best_result['model_name']}), vectorizer, and label encoder.")

# -------------------------------------------------------------------------------
if __name__ == '__main__':
    main()