import streamlit as st
from transformers import pipeline
import time

st.title("Sentiment Analysis App")

text = st.text_area("Enter text to analyze:", height=150)  # Larger text area

if st.button("Analyze"):
    if text:
        with st.spinner('Analyzing...'):
            try:
                classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
                result = classifier(text)
                labels = classifier.model.config.id2label
                predicted_label = labels[result[0]['label']]
                score = result[0]['score']

                mood = ""
                if 0 <= score < 0.25:
                    mood = "Unsure, but "
                elif 0.25 <= score < 0.5:
                    mood = "Somewhat "
                elif 0.5 <= score < 0.7:
                    mood = "Moderately "
                elif 0.7 <= score < 0.85:
                    mood = "Likely "
                elif 0.85 <= score < 0.95:
                    mood = "Strongly "
                else:
                    mood = "Definitely"
                mood = mood + predicted_label.capitalize()

                # Display results with styling
                st.write(f"**Sentiment:** {mood}")
                st.write(f"**Score:** {score:.2f}")  # Format score to 2 decimal places
                st.progress(score) #Progress bar for score
                if predicted_label == "positive":
                    st.success("Positive Sentiment Detected!")
                elif predicted_label == "negative":
                    st.error("Negative Sentiment Detected!")
                else:
                    st.info("Neutral Sentiment Detected.")
                with st.expander("See Advanced Metrics"):
                    st.json(result)
                    st.write("Here, the LABEL signifies the emotion and score signifies the strength of that emotion.\n LABEL_0 is negative, LABEL_1 is neutral and LABEL_2 is positive.\n The scores range from 0 to 1, is 0 being completely unsure and 1 being sure.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")