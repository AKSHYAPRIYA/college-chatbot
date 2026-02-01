import json
import random
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

with open('intents.json', 'r') as file:
    intents = json.load(file)

print("College Enquiry Chatbot 🤖")
print("Type 'quit' to exit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            print("Bot:", random.choice(intent['responses']))
            break
