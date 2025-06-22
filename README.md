# Smart Email Assistant

Smart Email Assistant is an intelligent system designed to automate email classification and response generation for departments like **HR**, **IT**, and **Other**. It uses machine learning and large language models (LLMs) to understand the email content, classify it into the correct category, generate a professional reply, and escalate unclear cases for manual review.

---

## Features

-  Classifies emails into HR, IT, or Other categories using a trained ML model
-  Generates professional responses using Hugging Face's FLAN-T5 language model
-  Escalates emails if the classifier is uncertain (confidence < 0.6)
-  Modular agent-based architecture (classifier, responder, escalator)
-  Interactive web UI built with Streamlit

---

##  Project Structure

smart-email-assistant/
├── app.py # Streamlit UI

├── orchestrator.py # Pipeline to connect all agents
├── models/ # Saved ML models (SVC, TFIDF, Label Encoder)
│ ├── model.pkl
│ ├── vectorizer.pkl
│ └── label_encoder.pkl
├── data/ # Datasets (original + cleaned)
│ └── cleaned_emails.csv
├── logs/ # Escalated emails stored as JSON
│ └── escalation_*.json
├── agents/ # Agents (Classifier, Responder, Escalator)
│ ├── email_classifier.py
│ ├── response_generator.py
│ └── escalation_agent.py
└── requirements.txt # Dependencies

