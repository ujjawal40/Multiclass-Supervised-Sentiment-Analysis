# ------------------------Libraries------------------------------------
import pandas as pd
import numpy as np
import string
import logging
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from joblib import dump, Parallel, delayed
import pickle
import sys
import os
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nickname = 'Fifi'


# ------------------------Custom Classes & Functions------------------------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess text by lowercasing and removing punctuation"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.str.lower().str.translate(str.maketrans('', '', string.punctuation))


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """Extract text statistics features using vectorized operations"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_words = X.str.split().str.len().fillna(0)
        avg_word_len = X.str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        num_unique = X.str.split().apply(lambda x: len(set(x)) if x else 0)
        num_upper = X.str.findall(r'\b[A-Z]{2,}\b').str.len()
        return np.column_stack([num_words, avg_word_len, num_unique, num_upper])


def clip_negatives(X):
    """Clip negative values in sparse/dense matrices"""
    if sparse.issparse(X):
        X.data = np.clip(X.data, a_min=0, a_max=None)
        return X
    return np.clip(X, 0, None)


# ------------------------Main Functions------------------------------------
def read_data(path):
    """Read and validate training data"""
    try:
        df = pd.read_excel(path)

        # Data validation
        if df[['Text', 'Label', 'split']].isnull().any().any():
            logger.warning("Missing values found. Dropping rows with missing values.")
            df = df.dropna(subset=['Text', 'Label', 'split'])

        if df.empty:
            logger.error("No data remaining after cleaning.")
            sys.exit(1)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        sys.exit(1)


def prepare_features():
    """Create feature union with preprocessing"""
    return FeatureUnion([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
        ('bow', CountVectorizer(ngram_range=(1, 2), max_features=10000)),
        ('char_ngrams', CountVectorizer(analyzer='char', ngram_range=(3, 5))),
        ('text_stats', TextStatsExtractor())
    ])


def build_pipeline(model):
    """Build complete processing pipeline"""
    return Pipeline([
        ('preprocess', TextPreprocessor()),
        ('features', prepare_features()),
        ('clip', FunctionTransformer(clip_negatives)),
        ('clf', model)
    ])


def tune_model(pipeline, X, y, param_grid):
    """Perform grid search with stratified cross-validation"""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    searcher = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro',
                            n_jobs=-1, verbose=1)
    searcher.fit(X, y)
    return searcher


def evaluate_model(model, X_test, y_test):
    """Generate comprehensive evaluation metrics"""
    preds = model.predict(X_test)
    logger.info("Classification Report:\n" + classification_report(y_test, preds))
    logger.info(f"Cohen's Kappa: {cohen_kappa_score(y_test, preds):.4f}")
    logger.info(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
    return 0.5 * cohen_kappa_score(y_test, preds) + 0.5 * f1_score(y_test, preds, average='macro')


# ------------------------Training Setup------------------------------------
def get_model_config(num_classes):
    """Return models and their parameter grids"""
    return [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {
                'clf__n_estimators': [200, 300],
                'clf__max_depth': [None, 30],
                'features__tfidf__max_features': [5000, 10000]
            }
        },
        {
            'name': 'XGBoost',
            'model': XGBClassifier(objective='multi:softmax',
                                   num_class=num_classes,
                                   eval_metric='mlogloss',
                                   random_state=42),
            'params': {
                'clf__n_estimators': [200, 300],
                'clf__learning_rate': [0.1, 0.05],
                'clf__max_depth': [6, 8]
            }
        }
    ]


# ------------------------Main Execution------------------------------------
def main():
    try:
        # Data preparation
        df = read_data("../Data/train.xlsx")
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']

        if train_df.empty or test_df.empty:
            logger.error("Invalid data split")
            sys.exit(1)

        # Label encoding
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['Label'])
        y_test = le.transform(test_df['Label'])

        # Model configuration
        model_configs = get_model_config(num_classes=len(le.classes_))

        best_score = -1
        best_model = None

        for config in model_configs:
            logger.info(f"\n{'=' * 40}\nTraining {config['name']}\n{'=' * 40}")

            # Build and tune pipeline
            pipeline = build_pipeline(config['model'])
            searcher = tune_model(pipeline, train_df['Text'], y_train, config['params'])

            # Evaluation
            test_score = evaluate_model(searcher.best_estimator_, test_df['Text'], y_test)

            if test_score > best_score:
                best_score = test_score
                best_model = searcher.best_estimator_
                logger.info(f"New best model: {config['name']} with score {test_score:.4f}")

        # Save best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"{nickname}_model_{timestamp}.pkl"
        with open(model_name, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Saved best model as {model_name}")

    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()