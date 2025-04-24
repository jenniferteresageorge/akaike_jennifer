import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')

class EmailClassifier:
    """Enhanced email classifier with better preprocessing and class balancing"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove special chars and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def _get_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    # In models.py, replace the train method with this version:
    def train(self, X, y):
    # Preprocess all texts
        X_processed = [self._preprocess_text(x) for x in X]
    
    # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
    
    # Get class weights
        class_weights = self._get_class_weights(y_encoded)
    
    # Individual models with class weights
    # Changed LinearSVC to SVC with probability=True
        from sklearn.svm import SVC
        svc = SVC(
            probability=True,  # This enables predict_proba
            class_weight=class_weights,
            kernel='linear',
            max_iter=10000
            )
    
        nb = MultinomialNB()
        lr = LogisticRegression(
            max_iter=1000,
            class_weight=class_weights
        )
        rf = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100
        )
    
        # Enhanced ensemble
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=30000, 
                ngram_range=(1, 3),
                sublinear_tf=True
            )),
            ('clf', VotingClassifier([
            ('svc', svc),
            ('nb', nb),
            ('lr', lr),
            ('rf', rf)
            ], voting='soft'))
        ])
    
        self.model.fit(X_processed, y_encoded)
    
    def predict(self, text: str) -> str:
        """Predict with preprocessing"""
        if not self.model:
            raise ValueError("Model not trained or loaded")
            
        processed_text = self._preprocess_text(text)
        encoded_pred = self.model.predict([processed_text])[0]
        return self.label_encoder.inverse_transform([encoded_pred])[0]
    
    
    def save_model(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.model:
            raise ValueError("Model not trained")
            
        joblib.dump(self.model, model_path)
        
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = joblib.load(model_path)
        self.classes = self.model.named_steps['clf'].classes_
    
    @staticmethod
    def load_data_from_csv(csv_path: str, text_col: str = "email", label_col: str = "type"):
        """
        Load data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            text_col: Name of the column containing email text
            label_col: Name of the column containing categories
            
        Returns:
            DataFrame with the loaded data
        """
        df = pd.read_csv(csv_path)
        return df[[text_col, label_col]].dropna()
    
    def train_from_csv(self, csv_path: str, text_col: str = "email", label_col: str = "type"):
        """
        Train the model using data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            text_col: Name of the column containing email text
            label_col: Name of the column containing categories
        """
        # Load data
        df = self.load_data_from_csv(csv_path, text_col, label_col)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col], df[label_col], test_size=0.2, random_state=42
        )
        
        # Train the model
        self.train(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self.save_model("email_classifier.joblib")
        print(f"Model trained and saved to email_classifier.joblib")