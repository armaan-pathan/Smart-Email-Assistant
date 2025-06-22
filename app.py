import streamlit as st
from agents.email_classifier import classify_email
from agents.response_generator import generate_response
from agents.escalation_agent import escalate_email

st.set_page_config(page_title="Smart Email Assistant 💼", layout="centered")
st.title(" Smart Email Assistant")
st.write("Automatically classify, respond to, or escalate emails.")

email_text = st.text_area(" Enter an email message:", height=150)

if st.button("Generate Reply"):
    if not email_text.strip():
        st.warning("Please enter an email.")
    else:
        try:
            classification = classify_email(email_text)
            category = classification["predicted_category"]
            confidence = classification["confidence"]

            st.markdown(f" Predicted Category: `{category}`")
            st.markdown(f" Confidence: `{round(confidence, 2)}`")

            if confidence < 0.6:
                escalation = escalate_email(
                    email_text=email_text,
                    predicted_category=category,
                    confidence=confidence,
                    reason="Low confidence - manual review required."
                )
                st.error(" Email escalated for manual review due to low confidence.")
                st.json(escalation)
            else:
                result = generate_response(email_text)
                st.success(" Generated Response:")
                st.write(result["response"])

        except Exception as e:
            st.error(f" An error occurred: {str(e)}")
