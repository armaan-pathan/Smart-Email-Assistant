from agents.email_classifier import classify_email
from agents.response_generator import generate_response
from agents.escalation_agent import escalate_email

def process_email(email_text):
    classification = classify_email(email_text)
    category = classification["predicted_category"]
    confidence = classification["confidence"]

    if confidence < 0.6:
        print("\n Confidence too low. Escalating...")
        escalation = escalate_email(
            email_text=email_text,
            predicted_category=category,
            confidence=confidence,
            reason="Low confidence - manual review required."
        )
        return escalation
    else:
        print("\n Generating professional response...")
        response_data = generate_response(email_text)
        return response_data

if __name__ == "__main__":
    print(" Smart Email Assistant (Orchestrator)")
    print("Type 'exit' to quit\n")

    while True:
        email = input(" Enter an email:\n> ")
        if email.lower() == 'exit':
            break

        try:
            result = process_email(email)
            print("\n Category:", result['predicted_category'])
            print(" Confidence:", result['confidence'])
            print(" Status:", result['status'])
            if result['status'] == 'responded':
                print(" Response:\n", result['response'])
            elif result['status'] == 'escalated':
                print(" Reason:", result['reason'])
        except Exception as e:
            print(" Error:", str(e))
