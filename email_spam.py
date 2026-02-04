#spam classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import re
df = pd.read_csv("spam_ham_dataset.csv")

def extract_features(text):
    """Extract useful features from text"""
    features = {}
    
    # Count features (these ARE useful!)
    features['has_url'] = 1 if re.search(r'http\S+|www\S+', text) else 0
    features['has_email'] = 1 if re.search(r'\S+@\S+', text) else 0
    features['num_exclamations'] = text.count('!')
    features['num_capitals'] = sum(1 for c in text if c.isupper())
    features['num_dollars'] = text.count('$')
    features['length'] = len(text)
    
    return features

def preprocess_text_smart(text):
    """
    Clean text but keep structure information
    DON'T remove everything - just normalize!
    """
    text = str(text).lower()
    
    # Replace newlines with space (but keep the space!)
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace URLs with token (keeps the information that URL exists!)
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    
    # Replace emails with token
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    
    # Replace numbers with token (but keep that there WAS a number)
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # Keep special characters initially - let's see what happens
    # We can remove them later if needed
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

# Apply smart preprocessing
df['text_clean'] = df['text'].apply(preprocess_text_smart)

# Show the difference
print("\n📝 Original:")
print(df['text'].iloc[4][:200])  # A spam email
print("\n📝 After smart preprocessing:")
print(df['text_clean'].iloc[4][:200])

# Use the cleaned text
x = df["text_clean"]
y = df["label_num"]


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42 , 
)

vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

print(x_train_vec.shape)

model = MultinomialNB()
model.fit(x_train_vec, y_train)
print("model trained")

y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy: {accuracy:.4f}, {accuracy*100:.2f}%")

print(classification_report(y_test, y_pred, target_names=['Ham',
                                                          'Spam']))
cm = confusion_matrix(y_test, y_pred)
print(cm)






