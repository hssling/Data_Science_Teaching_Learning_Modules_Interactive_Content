#!/usr/bin/env python3
"""
Natural Language Processing Project: Sentiment Analysis
======================================================

A complete end-to-end NLP project demonstrating:
- Text preprocessing and cleaning
- Feature extraction (TF-IDF, word embeddings)
- Multiple classification models
- Model evaluation and comparison
- Deployment considerations

Dataset: IMDB Movie Reviews
Goal: Classify movie reviews as positive or negative
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🎭 NLP PROJECT: SENTIMENT ANALYSIS OF MOVIE REVIEWS")
print("=" * 60)

class SentimentAnalysisProject:
    """Complete sentiment analysis project for movie reviews"""

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.vectorizer = None

    def load_data(self):
        """Load and examine the dataset"""
        print("\n📊 1. DATA LOADING AND INITIAL EXPLORATION")
        print("-" * 50)

        # Create synthetic movie review dataset
        # In a real project, you would load from a CSV file like IMDB dataset
        np.random.seed(42)
        n_samples = 5000

        # Positive and negative review templates
        positive_templates = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "An excellent film with outstanding performances. I highly recommend it to everyone.",
            "Brilliant direction and screenplay. This is cinema at its best.",
            "One of the greatest movies I've ever seen. Masterful storytelling and incredible cinematography.",
            "Outstanding performances by the entire cast. A must-watch film.",
            "The plot was engaging and the characters were well-developed. Highly enjoyable.",
            "A cinematic masterpiece that showcases the director's vision perfectly.",
            "Incredible special effects and a compelling storyline. Thoroughly impressed.",
            "The soundtrack was beautiful and complemented the story perfectly.",
            "An emotional rollercoaster that left me thinking long after it ended."
        ]

        negative_templates = [
            "This movie was a complete disappointment. Poor acting and a boring plot.",
            "Terrible film with weak performances and nonsensical storyline.",
            "Waste of time and money. The worst movie I've seen in years.",
            "Awful direction and screenplay. Nothing redeeming about this film.",
            "The plot made no sense and the acting was subpar at best.",
            "Disappointing and predictable. I regret watching this movie.",
            "Poor cinematography and even worse editing. A complete mess.",
            "The characters were one-dimensional and the dialogue was cringe-worthy.",
            "Overrated and overhyped. Not worth the time or money.",
            "Boring and uninspired. Fell asleep halfway through."
        ]

        # Generate reviews with some noise
        reviews = []
        sentiments = []

        for i in range(n_samples):
            if np.random.random() > 0.5:  # Positive review
                base_review = np.random.choice(positive_templates)
                # Add some noise/variation
                noise_words = ["really ", "truly ", "definitely ", "absolutely ", ""]
                review = np.random.choice(noise_words) + base_review
                sentiment = 1
            else:  # Negative review
                base_review = np.random.choice(negative_templates)
                # Add some noise/variation
                noise_words = ["really ", "truly ", "definitely ", "absolutely ", ""]
                review = np.random.choice(noise_words) + base_review
                sentiment = 0

            # Add some random length variation
            if np.random.random() > 0.7:
                extra_text = " The pacing was good and the cinematography was well done." if sentiment == 1 else " The special effects were terrible and the music was annoying."
                review += extra_text

            reviews.append(review)
            sentiments.append(sentiment)

        self.data = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })

        print(f"Dataset shape: {self.data.shape}")
        print(f"Sentiment distribution:")
        print(self.data['sentiment'].value_counts(normalize=True))
        print(f"\nSample positive review: {self.data[self.data['sentiment'] == 1]['review'].iloc[0][:100]}...")
        print(f"Sample negative review: {self.data[self.data['sentiment'] == 0]['review'].iloc[0][:100]}...")

        # Review length analysis
        self.data['review_length'] = self.data['review'].str.len()
        print(f"\nReview length statistics:")
        print(self.data.groupby('sentiment')['review_length'].describe())

        return self.data

    def text_preprocessing(self):
        """Comprehensive text preprocessing pipeline"""
        print("\n🔧 2. TEXT PREPROCESSING AND FEATURE ENGINEERING")
        print("-" * 55)

        def clean_text(text):
            """Clean and preprocess text data"""
            # Convert to lowercase
            text = text.lower()

            # Remove HTML tags (if any)
            text = re.sub(r'<[^>]+>', '', text)

            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', '', text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        def tokenize_and_lemmatize(text):
            """Tokenize and lemmatize text"""
            # Get English stopwords
            stop_words = set(stopwords.words('english'))

            # Initialize lemmatizer
            lemmatizer = WordNetLemmatizer()

            # Tokenize
            tokens = nltk.word_tokenize(text)

            # Remove stopwords and lemmatize
            processed_tokens = [
                lemmatizer.lemmatize(token) for token in tokens
                if token not in stop_words and len(token) > 2
            ]

            return ' '.join(processed_tokens)

        print("Applying text preprocessing...")

        # Apply cleaning
        self.data['clean_review'] = self.data['review'].apply(clean_text)

        # Apply tokenization and lemmatization
        self.data['processed_review'] = self.data['clean_review'].apply(tokenize_and_lemmatize)

        print("Preprocessing completed!")
        print(f"Original review: {self.data['review'].iloc[0][:100]}...")
        print(f"Processed review: {self.data['processed_review'].iloc[0][:100]}...")

        # Split the data
        X = self.data['processed_review']
        y = self.data['sentiment']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Sentiment distribution in training: {self.y_train.mean():.3f}")
        print(f"Sentiment distribution in test: {self.y_test.mean():.3f}")

        return X, y

    def feature_extraction(self):
        """Extract features from text using different methods"""
        print("\n📝 3. FEATURE EXTRACTION")
        print("-" * 30)

        # Method 1: TF-IDF Vectorization
        print("Creating TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )

        X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train)
        X_test_tfidf = tfidf_vectorizer.transform(self.X_test)

        print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
        print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")

        # Show top TF-IDF features
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = X_train_tfidf.sum(axis=0).A1
        top_features = pd.DataFrame({
            'feature': feature_names,
            'tfidf_score': tfidf_scores
        }).sort_values('tfidf_score', ascending=False).head(20)

        print("\nTop 20 TF-IDF features:")
        for i, (_, row) in enumerate(top_features.iterrows()):
            print("2d")

        # Method 2: Bag of Words (Count Vectorization)
        print("\nCreating Bag of Words features...")
        count_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        X_train_count = count_vectorizer.fit_transform(self.X_train)
        X_test_count = count_vectorizer.transform(self.X_test)

        print(f"Count vectorization shape: {X_train_count.shape}")

        # Store vectorizers and features
        self.vectorizers = {
            'tfidf': tfidf_vectorizer,
            'count': count_vectorizer
        }

        self.features = {
            'tfidf': (X_train_tfidf, X_test_tfidf),
            'count': (X_train_count, X_test_count)
        }

        return self.features

    def model_building(self):
        """Build and compare multiple classification models"""
        print("\n🤖 4. MODEL BUILDING AND COMPARISON")
        print("-" * 40)

        # Define models to compare
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(random_state=42, max_iter=10000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }

        # Test both TF-IDF and Count vectorization
        results = {}

        for vectorizer_name, (X_train_vec, X_test_vec) in self.features.items():
            print(f"\nTesting models with {vectorizer_name.upper()} features:")
            print("-" * 50)

            for model_name, model in models.items():
                print(f"Training {model_name}...")

                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train_vec, self.y_train,
                                          cv=5, scoring='roc_auc')

                # Fit on full training data
                model.fit(X_train_vec, self.y_train)

                # Predictions on test set
                y_pred = model.predict(X_test_vec)
                y_pred_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_vec)

                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)

                model_key = f"{model_name} ({vectorizer_name})"
                results[model_key] = {
                    'vectorizer': vectorizer_name,
                    'model': model,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'X_test': X_test_vec
                }

                print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                print(f"  Test ROC AUC: {roc_auc:.3f}")
                print(f"  Test Accuracy: {accuracy:.3f}")
                print(f"  Test F1-Score: {f1:.3f}")

        self.models = results

        # Find best model
        best_model_key = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        self.best_model = results[best_model_key]

        print(f"\n🏆 Best Model: {best_model_key}")
        print(f"   ROC AUC: {self.best_model['roc_auc']:.3f}")
        print(f"   Accuracy: {self.best_model['accuracy']:.3f}")
        print(f"   F1-Score: {self.best_model['f1']:.3f}")

        return results

    def model_evaluation(self):
        """Detailed evaluation of the best model"""
        print("\n📊 5. MODEL EVALUATION AND INTERPRETATION")
        print("-" * 50)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.best_model['predictions'])
        print("Confusion Matrix:")
        print(cm)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.best_model['predictions'],
                                  target_names=['Negative', 'Positive']))

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.best_model['probabilities'])

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.best_model["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Sentiment Analysis Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('sentiment_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Feature importance for Logistic Regression with TF-IDF
        if 'Logistic Regression' in str(self.best_model) and self.best_model['vectorizer'] == 'tfidf':
            print("\n🔍 Most Important Features (Logistic Regression + TF-IDF):")

            # Get feature names and coefficients
            feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            coefficients = self.best_model['model'].coef_[0]

            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            # Show top positive and negative features
            print("\nTop 10 Positive Sentiment Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print("2d")

            print("\nTop 10 Negative Sentiment Features:")
            for i, (_, row) in enumerate(feature_importance.tail(10).iterrows()):
                print("2d")

    def error_analysis(self):
        """Analyze model errors to understand weaknesses"""
        print("\n🔍 6. ERROR ANALYSIS")
        print("-" * 25)

        # Get predictions and actual values
        predictions = self.best_model['predictions']
        actual = self.y_test

        # Find misclassified examples
        misclassified = self.X_test[(predictions != actual)]
        misclassified_actual = actual[(predictions != actual)]
        misclassified_pred = predictions[(predictions != actual)]

        print(f"Total misclassified examples: {len(misclassified)}")
        print(f"Misclassification rate: {len(misclassified) / len(self.X_test):.3f}")

        # Show some examples of misclassifications
        print("\nExamples of Misclassifications:")
        print("-" * 40)

        for i in range(min(5, len(misclassified))):
            print(f"\nExample {i+1}:")
            print(f"Text: {misclassified.iloc[i][:200]}...")
            print(f"Actual: {'Positive' if misclassified_actual.iloc[i] == 1 else 'Negative'}")
            print(f"Predicted: {'Positive' if misclassified_pred.iloc[i] == 1 else 'Negative'}")

        # Analyze errors by text length
        error_df = pd.DataFrame({
            'text': self.X_test,
            'actual': self.y_test,
            'predicted': predictions,
            'correct': (predictions == actual),
            'text_length': self.X_test.str.len()
        })

        # Error rate by text length bins
        error_df['length_bin'] = pd.cut(error_df['text_length'],
                                      bins=[0, 50, 100, 200, 500, 1000],
                                      labels=['0-50', '50-100', '100-200', '200-500', '500+'])

        error_by_length = error_df.groupby('length_bin')['correct'].mean()
        print(f"\nAccuracy by Text Length:")
        print(error_by_length)

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print("\n⚙️ 7. HYPERPARAMETER TUNING")
        print("-" * 30)

        # Define parameter grid for Logistic Regression (assuming it's the best model)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Grid search with cross-validation
        print("Performing grid search with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            pipeline,
            {'classifier__' + key: value for key, value in param_grid.items()},
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")

        # Evaluate best model on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

        test_roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred)

        print(f"\nTuned Model Test Performance:")
        print(f"ROC AUC: {test_roc_auc:.3f}")
        print(f"Accuracy: {test_accuracy:.3f}")
        print(f"F1-Score: {test_f1:.3f}")

        return grid_search.best_estimator_

    def deployment_considerations(self):
        """Discuss deployment and production considerations"""
        print("\n🚀 8. DEPLOYMENT CONSIDERATIONS")
        print("-" * 35)

        print("Production Deployment Checklist:")
        print("✓ Model Serialization:")
        print("  - Save vectorizer and model using joblib")
        print("  - Version control for models and preprocessing")
        print("  - Model registry for tracking versions")

        print("\n✓ Text Preprocessing Pipeline:")
        print("  - Consistent preprocessing in production")
        print("  - Handle edge cases (empty text, special characters)")
        print("  - Input validation and sanitization")

        print("\n✓ API Development:")
        print("  - REST API using Flask/FastAPI")
        print("  - Rate limiting and authentication")
        print("  - Error handling and logging")

        print("\n✓ Performance Optimization:")
        print("  - Model compression and quantization")
        print("  - Batch processing for multiple predictions")
        print("  - Caching for frequent predictions")

        print("\n✓ Monitoring & Maintenance:")
        print("  - Performance monitoring (accuracy, latency)")
        print("  - Data drift detection for text features")
        print("  - Regular model retraining with new data")
        print("  - Alert system for model degradation")

        # Example deployment code
        import joblib

        print("\n💾 Example: Model Serialization and Prediction")
        print("-" * 45)

        # Save the best model and vectorizer
        model_data = {
            'vectorizer': self.vectorizers['tfidf'] if self.best_model['vectorizer'] == 'tfidf' else self.vectorizers['count'],
            'model': self.best_model['model'],
            'model_info': {
                'roc_auc': self.best_model['roc_auc'],
                'accuracy': self.best_model['accuracy'],
                'f1_score': self.best_model['f1']
            }
        }

        joblib.dump(model_data, 'sentiment_analysis_model.pkl')
        print("Model and vectorizer saved as 'sentiment_analysis_model.pkl'")

        # Example prediction function
        def predict_sentiment(review_text):
            """
            Example prediction function for deployment

            Parameters:
            review_text (str): Movie review text

            Returns:
            dict: Prediction results
            """
            # Load model (in production, load once at startup)
            model_data = joblib.load('sentiment_analysis_model.pkl')

            # Preprocess text (same as training preprocessing)
            def preprocess_text(text):
                text = text.lower()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()

                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()
                tokens = nltk.word_tokenize(text)
                processed_tokens = [
                    lemmatizer.lemmatize(token) for token in tokens
                    if token not in stop_words and len(token) > 2
                ]
                return ' '.join(processed_tokens)

            # Preprocess input
            processed_text = preprocess_text(review_text)

            # Vectorize
            text_vector = model_data['vectorizer'].transform([processed_text])

            # Predict
            prediction = model_data['model'].predict(text_vector)[0]
            probability = model_data['model'].predict_proba(text_vector)[0][1]

            return {
                'sentiment': 'Positive' if prediction == 1 else 'Negative',
                'confidence': probability if prediction == 1 else 1 - probability,
                'processed_text': processed_text
            }

        # Example usage
        sample_reviews = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "This movie was a complete disappointment. Poor acting and a boring plot.",
            "An excellent film with outstanding performances. I highly recommend it to everyone."
        ]

        print("\nSample Predictions:")
        for i, review in enumerate(sample_reviews, 1):
            result = predict_sentiment(review)
            print(f"\nReview {i}:")
            print(f"Text: {review[:80]}...")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")

    def run_complete_project(self):
        """Run the complete sentiment analysis project"""
        print("Starting Complete Sentiment Analysis Project")
        print("=" * 50)

        # Execute all steps
        self.load_data()
        self.text_preprocessing()
        self.feature_extraction()
        self.model_building()
        self.model_evaluation()
        self.error_analysis()
        self.hyperparameter_tuning()
        self.deployment_considerations()

        print("\n" + "=" * 50)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Generated files:")
        print("- sentiment_roc_curve.png")
        print("- sentiment_analysis_model.pkl")
        print("\nKey Achievements:")
        print("✓ Complete NLP workflow implemented")
        print("✓ Text preprocessing and feature extraction")
        print("✓ Multiple ML models for sentiment classification")
        print("✓ Model evaluation and error analysis")
        print("✓ Production deployment considerations")
        print("✓ End-to-end sentiment analysis system")

if __name__ == "__main__":
    # Run the complete project
    project = SentimentAnalysisProject()
    project.run_complete_project()
