import pandas as pd
import numpy as np
import os
import re
import joblib
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

nltk.download('stopwords')
nltk.download('wordnet')


class EmailClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    
    def preprocess(self, text: str) -> str:
        text = text.lower()

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Keep alphanumerics, dots, underscores, hyphens (useful in tech terms)
        text = re.sub(r'[^a-zA-Z0-9\s._-]', ' ', text)

        tokens = text.split()

        # Custom stopwords: remove common words but retain useful ones
        custom_stop_words = self.stop_words - {'no', 'not', 'nor', 'against', 'aren', "aren't", 'isn', "isn't"}
    
        # Lemmatize and filter
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in custom_stop_words and len(word) > 1
        ]

        return ' '.join(tokens)



    def train(self, X, y, use_grid_search=False):
        print("Preprocessing data...")
        X_processed = [self.preprocess(text) for text in X]

        print("Oversampling minority classes...")
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(np.array(X_processed).reshape(-1, 1), y)
        X_resampled = X_resampled.ravel()  # Flatten the array back

        print("Initializing pipeline...")
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                max_features=15000,
                ngram_range=(1, 3),
                sublinear_tf=True
            )),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42))
        ])


        if use_grid_search:
            print("Running Grid Search...")
            params = {
                'clf__alpha': [1e-4, 1e-3, 1e-2],
                'clf__penalty': ['l2', 'l1', 'elasticnet']
            }
            grid = GridSearchCV(pipeline, param_grid=params, scoring='f1_weighted', cv=5, verbose=2)
            grid.fit(X_resampled, y_resampled)
            self.model = grid.best_estimator_
            print("Best Params:", grid.best_params_)
        else:
            print("Fitting model...")
            pipeline.fit(X_resampled, y_resampled)
            self.model = pipeline

        print("Model trained.")
        self.classes = self.model.named_steps['clf'].classes_


    def predict(self, text: str) -> str:
        if not self.model:
            raise ValueError("Model not trained or loaded")
        processed_text = self.preprocess(text)
        return self.model.predict([processed_text])[0]

    def save_model(self, model_path: str):
        if not self.model:
            raise ValueError("Model not trained")
        joblib.dump(self.model, model_path)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        self.classes = self.model.named_steps['clf'].classes_

    @staticmethod
    def load_data_from_csv(csv_path: str, text_col: str = "email", label_col: str = "type"):
        df = pd.read_csv(csv_path)
        return df[[text_col, label_col]].dropna()

    def train_from_csv(self, csv_path: str, text_col: str = "email", label_col: str = "type", use_grid_search=False):
        df = self.load_data_from_csv(csv_path, text_col, label_col)
        #X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
        )
        self.train(X_train, y_train, use_grid_search=use_grid_search)
        X_test_processed = [self.preprocess(text) for text in X_test]
        y_pred = self.model.predict(X_test_processed)
        print(classification_report(y_test, y_pred))
        self.save_model("email_classifier.joblib")
        print("Model trained and saved to email_classifier.joblib")
