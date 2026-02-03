from flask import Flask, render_template, request, jsonify, session
import pickle
import json
import random
import datetime

app = Flask(__name__)
app.secret_key = "college_chatbot_secret"

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load data files
with open('intents.json') as file:
    intents = json.load(file)

with open('college_data.json') as file:
    college_data = json.load(file)

with open('faq.json') as file:
    faq = json.load(file)


# -------- Greeting based on time --------
def time_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning ☀️"
    elif hour < 18:
        return "Good Afternoon 🌤️"
    else:
        return "Good Evening 🌙"


# -------- FAQ check --------
def check_faq(message):
    for key in faq:
        if key in message:
            return faq[key]
    return None


# -------- College data check --------
def check_college_data(message):
    if "principal" in message:
        return f"Our principal is {college_data['principal']}."
    if "timing" in message:
        return f"College timing is {college_data['college_timing']}."
    if "library" in message:
        return f"Library timing is {college_data['library_timing']}."
    if "email" in message:
        return f"You can email us at {college_data['email']}."
    if "address" in message or "location" in message:
        return f"Our college is located at {college_data['address']}."
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower().strip()

    # -------- Session memory --------
    if "history" not in session:
        session["history"] = []
    session["history"].append(user_message)

    # 1️⃣ Greeting & Time-aware greetings
    greetings = ["hi", "hello", "hey"]
    morning_words = ["good morning"]
    afternoon_words = ["good afternoon"]
    evening_words = ["good evening"]
    bye_words = ["bye", "goodbye", "see you", "exit"]

    if user_message in greetings:
        return jsonify({"reply": f"{time_greeting()} 😊 How can I help you today?"})

    if user_message in morning_words + afternoon_words + evening_words:
        return jsonify({"reply": f"{time_greeting()} 😊 How can I help you today?"})

    if user_message in bye_words:
        return jsonify({"reply": "Goodbye! Have a great day 👋"})

    # 2️⃣ FAQ reply
    faq_reply = check_faq(user_message)
    if faq_reply:
        return jsonify({"reply": faq_reply})

    # 3️⃣ College data reply
    college_reply = check_college_data(user_message)
    if college_reply:
        return jsonify({"reply": college_reply})

    # 4️⃣ Normalize text
    if "fees" in user_message:
        user_message = user_message.replace("fees", "fee")

    # 5️⃣ ML model with confidence check
    X_test = vectorizer.transform([user_message])
    probs = model.predict_proba(X_test)[0]
    max_prob = max(probs)

    if max_prob < 0.35:
        return jsonify({
            "reply": (
                "I'm not sure I understood 🤔\n\n"
                "You can ask about:\n"
                "• Courses\n• Fees\n• Hostel\n• Admission\n• Contact\n• Principal\n• Timing"
            )
        })

    predicted_tag = model.classes_[probs.argmax()]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return jsonify({"reply": random.choice(intent["responses"])})

    # 6️⃣ Final fallback
    return jsonify({"reply": "Please ask questions related to the college."})


if __name__ == "__main__":
    app.run(debug=True)
