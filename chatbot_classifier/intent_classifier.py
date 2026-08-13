"""
Intent Classifier Module
Trains and evaluates a sentence classifier for intent detection
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class IntentClassifier:
    """Classifies user input into predefined intents"""
    
    def __init__(self, model_path='models/intent_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.intents = None
        
    def create_training_data(self):
        """Create dummy training data for intent classification"""
        training_data = [
            # Greeting
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("hey", "greeting"),
            ("good morning", "greeting"),
            ("hey buddy", "greeting"),
            
            # Goodbye
            ("bye", "goodbye"),
            ("goodbye", "goodbye"),
            ("see you later", "goodbye"),
            ("talk to you soon", "goodbye"),
            ("farewell", "goodbye"),
            
            # Help
            ("help me", "help"),
            ("i need help", "help"),
            ("can you assist me", "help"),
            ("what can you do", "help"),
            ("how do you work", "help"),
            
            # Order
            ("i want to place an order", "order"),
            ("can i buy something", "order"),
            ("i would like to purchase", "order"),
            ("show me products", "order"),
            ("what products do you have", "order"),
            
            # Support
            ("my order is late", "support"),
            ("i have an issue", "support"),
            ("this is broken", "support"),
            ("i need customer service", "support"),
            ("there is a problem", "support"),
            
            # Information
            ("tell me about your company", "information"),
            ("what is your address", "information"),
            ("do you have a store", "information"),
            ("what are your hours", "information"),
            ("how can i contact you", "information"),
        ]
        return training_data
    
    def train(self, training_data=None):
        """Train the intent classifier"""
        if training_data is None:
            training_data = self.create_training_data()
        
        sentences = [data[0] for data in training_data]
        labels = [data[1] for data in training_data]
        self.intents = list(set(labels))
        
        # Create pipeline with TF-IDF and Random Forest
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100, lowercase=True, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        self.model.fit(sentences, labels)
        print(f"Model trained on {len(training_data)} samples")
        print(f"Intents learned: {self.intents}")
        
        return self
    
    def predict(self, text, confidence_threshold=0.3):
        """Predict intent and confidence"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        intent = self.model.predict([text])[0]
        
        # Get confidence scores
        proba = self.model.predict_proba([text])[0]
        confidence = max(proba)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'below_threshold': confidence < confidence_threshold
        }
    
    def save(self):
        """Save model to disk"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            pickle.dump(self.intents, f)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.intents = pickle.load(f)
        print(f"Model loaded from {self.model_path}")
        return self
    
    def evaluate(self):
        """Evaluate model on test data"""
        training_data = self.create_training_data()
        sentences = [data[0] for data in training_data]
        labels = [data[1] for data in training_data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=0.2, random_state=42
        )
        
        self.train([(s, l) for s, l in zip(X_train, y_train)])
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
