import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
from agents.escalation_agent import escalate_email


# Add path to import your email classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.email_classifier import classify_email

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def generate_response(email_text):
    classification = classify_email(email_text)
    category = classification["predicted_category"]
    confidence = classification["confidence"]

    if confidence < 0.6:
        escalation = escalate_email(email_text, category, confidence, reason="Low confidence - manual review required.")
        return escalation

    
    prompt = (
        f"You are an assistant in the {category} department.\n"
        f"Your job is to write polite, helpful, and professional email responses.\n"
        f"Reply to the following email message:\n"
        f"\"{email_text}\"\n\n"
        f"Make sure the response is formal, respectful, and complete."
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return {
        "email_text": email_text,
        "predicted_category": category,
        "confidence": float(round(confidence, 2)),
        "status": "responded",
        "response": response
    }

if __name__ == "__main__":
    while True:
        email = input("\n Enter email text (or type 'exit' to quit):\n> ")
        if email.lower() == 'exit':
            break
        try:
            output = generate_response(email)
            print("\n Predicted Category:", output['predicted_category'])
            print(" Confidence:", output['confidence'])
            print(" Status:", output['status'])
            if output['status'] == 'responded':
                print(" Response:\n", output['response'])
            else:
                print(" Escalation Reason:", output['reason'])
        except Exception as e:
            print(" Error:", str(e))
