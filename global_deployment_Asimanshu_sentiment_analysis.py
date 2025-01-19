import streamlit as st
from transformers import pipeline
import re

st.set_page_config(page_title="Sentiment Analysis App", page_icon=":smiley:", layout="wide")

st.title("Sentiment Analysis App")
st.write("This app analyzes the sentiment of text using a pre-trained model.")

text = st.text_area("Enter text to analyze:", height=150, placeholder="Type your text here...")

# Model loading outside the button click for efficiency
try:
    @st.cache_resource  # Cache the model to avoid reloading on every run
    def load_model():
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    classifier = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}. Please check your internet connection and try again.")
    st.stop() #Stop execution if the model does not load

if st.button("Analyze"):
    if not text.strip():  # Check for empty or whitespace-only input
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            try:
                result = classifier(text)
                labels = classifier.model.config.id2label
                predicted_label = labels[result[0]['label']].lower() #lower case for consistency
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
                    mood = "Definitely "
                mood += predicted_label.capitalize()

                col1, col2 = st.columns(2) #Columns for layout
                with col1:
                    st.metric(label="Sentiment", value=mood)
                    st.metric(label="Score", value=f"{score:.2f}")
                    st.progress(score)
                    if predicted_label == "positive":
                        st.success("Positive Sentiment Detected! :thumbsup:")
                    elif predicted_label == "negative":
                        st.error("Negative Sentiment Detected! :thumbsdown:")
                    else:
                        st.info("Neutral Sentiment Detected. :neutral_face:")
                with col2:
                    with st.expander("See Advanced Metrics"):
                        st.json(result)
                        st.write("Here, the LABEL signifies the emotion and score signifies the strength of that emotion.\n LABEL_0 is negative, LABEL_1 is neutral and LABEL_2 is positive.\n The scores range from 0 to 1, with 0 being completely unsure and 1 being certain.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Enter text above and click 'Analyze' to begin.")

st.markdown("---")