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

## 🧠 Project Architecture

      +--------------------------+
      |      User Input Email    |
      +------------+-------------+
                   |
        ┌──────────▼──────────┐
        │  Email Classifier   │  ← ML model classifies email (HR/IT/Other)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────┐
 High Confidence (>= 0.6)         │
        │                         │
┌───────────▼──────────┐ ┌─────────▼─────────────┐
│ Response Generator │ │ Escalation Agent │ ← If low confidence
│ (LLM-based) │ │ Logs escalation cases │
└──────────┬───────────┘ └───────────────────────┘
│
▼
Final Email Response

---


## ML Model Details & Evaluation

- **Model Type**: Support Vector Classifier (SVC)
- **Text Vectorization**: TF-IDF Vectorizer
- **Label Encoding**: Scikit-learn's LabelEncoder

### Dataset:
- 300 emails (100 HR, 100 IT, 100 Other)
- After cleaning & deduplication: **91 emails used for training**

### Evaluation Metrics:
| Metric       | Score     |
|--------------|-----------|
| Accuracy     | ~90%+     |
| Cross-Validation | 5-Fold |
| Evaluation Done On | Cleaned dataset |

> Models are saved as:
- `models/model.pkl`
- `models/vectorizer.pkl`
- `models/label_encoder.pkl`

---

## 💬 Prompt Design & LLM Integration

- **Model Used**: `google/flan-t5-base` from Hugging Face
- **Library**: `transformers`

---

##  Setup Instructions

### 1. Clone the Repository
git clone https://github.com/your-username/smart-email-assistant.git
cd smart-email-assistant

### 2. Install Dependencies
pip install -r requirements.txt

### 3.  Requirements File
streamlit
scikit-learn
pandas
joblib
transformers
torch

---

### How to Run
### Streamlit Web App
streamlit run app.py

### Terminal (CLI) Version
python orchestrator.py

---

##  Project Structure

smart-email-assistant/
├── app.py                    # Streamlit UI
├── orchestrator.py           # CLI Orchestrator
├── models/                   # Saved ML models
├── data/                     # Dataset files
├── logs/                     # Escalated emails in JSON
├── agents/                   # Modular ML + LLM agents
│   ├── email_classifier.py
│   ├── response_generator.py
│   └── escalation_agent.py
├── requirements.txt
└── README.md


