from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load 3-class dataset from Hugging Face
print("📦 Loading TweetEval 3-class Sentiment dataset...")
dataset = load_dataset("tweet_eval", "sentiment")

# Convert to pandas DataFrames
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Features and labels
X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

# TF-IDF Vectorizer
print("🔠 Extracting text features...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
print("🤖 Training Logistic Regression model (3-class)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
print("📊 Evaluating model...")
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Neutral","Positive"]))

# Save Model and Vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/sentiment3_model.pkl")
joblib.dump(vectorizer, "model/vectorizer3.pkl")

print("✅ 3-class sentiment model saved to ./model/")
