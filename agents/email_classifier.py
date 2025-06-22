import joblib
import string

# Load model, vectorizer, and label encoder
clf = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def classify_email(email_text):
    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])

    # Predict category probabilities
    proba = clf.predict_proba(vec)[0]
    pred_index = proba.argmax()
    confidence = proba[pred_index]
    category = label_encoder.inverse_transform([pred_index])[0]

    return {
        "email_text": email_text,
        "predicted_category": category,
        "confidence": float(round(confidence, 2))
    }

if __name__ == "__main__":
    while True:
        email = input("\nType an email (or type 'exit' to quit):\n> ")
        if email.lower() == 'exit':
            print("Exiting...")
            break
        result = classify_email(email)
        print("\nPrediction:")
        print(result)