import os
import json
from datetime import datetime

ESCALATION_LOG_DIR = "logs"
os.makedirs(ESCALATION_LOG_DIR, exist_ok=True)

def escalate_email(email_text, predicted_category, confidence, reason="Low confidence"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_entry = {
        "timestamp": timestamp,
        "email_text": email_text,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "reason": reason,
        "status": "escalated"
    }

    log_file = os.path.join(ESCALATION_LOG_DIR, f"escalation_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=4)

    print(f"\n Email has been escalated and logged to: {log_file}")
    return log_entry
