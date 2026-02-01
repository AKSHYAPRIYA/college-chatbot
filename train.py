import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents file
with open('intents.json', 'r') as file:
    data = json.load(file)

texts = []
labels = []

# Prepare training data
for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# Convert text to numerical form
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(texts)

# Train the model
model = LogisticRegression()
model.fit(X, labels)

# Save trained model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Training completed successfully!")
