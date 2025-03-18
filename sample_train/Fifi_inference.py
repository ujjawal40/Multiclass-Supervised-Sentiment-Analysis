import argparse
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import pickle
from joblib import load
import numpy as np



nickname = 'Fifi'
# -------------------------------------------------------------------------------
def read_data(path):
    df = pd.read_excel(path)
    return df


def split_data(df, split='test'):
    test_df = df[df['split'] == split]
    return test_df


def load_model(name):
    with open(name, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


def load_labelencoder(file_name):
    return load(file_name)


def inference(test_df, pipeline, label_encoder):
    # Use the pipeline directly for prediction, which includes the vectorizer
    predictions = pipeline.predict(test_df['Text'])
    # Ensure label encoder is fitted; if not, fit it on test labels as a fallback
    if not hasattr(label_encoder, 'classes_'):
        print("LabelEncoder not fitted. Fitting on test labels as a fallback.")
        label_encoder.fit(test_df['Label'])
    predictions = label_encoder.inverse_transform(predictions)
    return predictions


def metrics(predictions, test_df):
    kappa = cohen_kappa_score(test_df['Label'], predictions)
    f1 = f1_score(test_df['Label'], predictions, average='macro')
    return 0.5 * kappa + 0.5 * f1


# ------------------------Main Function------------------------------------
def main(data_path, train_df, model_path, split):
    # Load and split data
    df = read_data(data_path + train_df)
    test_df = split_data(df, split)

    # Load the label encoder
    label_encoder = load_labelencoder(file_name=f'{model_path}/{nickname}_labelencoder.joblib')

    # Load the entire pipeline (includes vectorizer and classifier)
    pipeline = load_model(f'{model_path}/{nickname}_model.pickle')

    # Run inference
    predictions = inference(test_df, pipeline, label_encoder)

    # Save predictions
    np.save(f'{model_path}/{nickname}-{split}_predictions.npy', predictions)

    # Compute and print metric score
    metric_score = metrics(predictions, test_df)
    print(f'Metric score: {metric_score}')


# -------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to data file')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--train_df', type=str, help='train_df')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--split', type=str, help='Split sample')
    args = parser.parse_args()

    main(args.data_path, args.train_df, args.model_path, args.split)