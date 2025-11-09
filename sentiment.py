import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# STEP 1: LOAD THE SENTIMENT140 DATASET
# -------------------------------
# The dataset has no headers, so we define them manually
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('sentiment140.csv', encoding='latin-1', names=columns)

# Keep only the relevant columns
df = df[['target', 'text']]

# Map 0 -> Negative, 4 -> Positive (2 = Neutral sometimes)
df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})
df = df.dropna(subset=['text'])
df = df.sample(50000, random_state=42)  # sample for faster testing

print(df.head())

# -------------------------------
# STEP 2: TEXT CLEANING
# -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())  # keep only letters
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# -------------------------------
# STEP 3: TF-IDF FEATURES
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# -------------------------------
# STEP 4: TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# STEP 5: MODEL TRAINING
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# STEP 6: EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['Positive','Negative'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Positive','Negative'], yticklabels=['Positive','Negative'])
plt.title('Confusion Matrix')
plt.show()

# -------------------------------
# STEP 7: SAVE MODEL
# -------------------------------
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
print("✅ Model and vectorizer saved to ./model/")

# -------------------------------
# STEP 8: TEST WITH CUSTOM INPUT
# -------------------------------
def predict_sentiment(text):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    return model.predict(vec)[0]

print("Example predictions:")
print("I love this product! ->", predict_sentiment("I love this product!"))
print("This movie was terrible. ->", predict_sentiment("This movie was terrible."))
