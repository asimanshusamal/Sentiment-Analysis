import streamlit as st
from transformers import pipeline
import re

# Function to load the sentiment analysis model
def load_model():
  try:
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
  except Exception as e:
    st.error(f"Error loading the model: {e}. Please check your internet connection and try again.")
    st.stop()  # Stop execution if the model fails to load

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":smiley:", layout="wide")

# App title and description
st.title("Sentiment Analysis App")
st.write("This app analyzes the sentiment of text using a pre-trained model.")

# Text input area
text = st.text_area("Enter text to analyze:", height=150, placeholder="Type your text here...")

# Load the model outside the button click for efficiency
classifier = load_model()

# Analyze button
if st.button("Analyze"):
  # Check for empty or whitespace-only input
  if not text.strip():
    st.warning("Please enter some text to analyze.")
  else:
    with st.spinner('Analyzing...'):
      try:
        result = classifier(text)
        labels = classifier.model.config.id2label
        predicted_label = labels[result[0]['label']].lower()  # Lowercase for consistency
        score = result[0]['score']

        # Define sentiment mood based on score
        mood = {
            0: "Unsure, but ",
            0.25: "Somewhat ",
            0.5: "Moderately ",
            0.7: "Likely ",
            0.85: "Strongly ",
            0.95: "Definitely "
        }.get(score, "")
        mood += predicted_label.capitalize()

        # Display results in columns
        col1, col2 = st.columns(2)
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

# App footer
st.markdown("---")
