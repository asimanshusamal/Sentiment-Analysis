import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis App")

text = st.text_area("Enter text to analyze:", height=150)

@st.cache_data  # Correctly using st.cache_data
def loadmod():
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment") #Default model was mediocre, I used this one trained on twitter messages
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

classifier = load_mod()

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.spinner('Analyzing...')
        try:
            result = classifier(text)
            predicted_label = result[0]['label']
            score = result[0]['score']

            if predicted_label == "LABEL_2":
                sentiment = "Positive"
            elif predicted_label == "LABEL_1":
                sentiment = "Neutral"
            elif predicted_label == "LABEL_0":
                sentiment = "Negative"
            else:
                sentiment = "Unknown"
                st.error("Unexpected label returned by the model.")

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
            mood += sentiment

            st.metric(f"Sentiment: {mood}")
            st.metric(f"Score: {score:.2f}")
            st.progress(score)

            if sentiment == "Positive":
                st.success("Positive Sentiment Detected! :thumbsup:")
            elif sentiment == "Negative":
                st.error("Negative Sentiment Detected! :thumbsdown:")
            else:
                st.info("Neutral Sentiment Detected. :neutral_face:")

            st.write("See Advanced Metrics:")
            st.json(result)
            st.write("Here, the LABEL signifies the emotion and score signifies the strength of that emotion.")
            st.write("LABEL_0 is negative, LABEL_1 is neutral and LABEL_2 is positive.")
            st.write("The scores range from 0 to 1, with 0 being completely unsure and 1 being certain.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Enter text above and click 'Analyze' to begin.")

st.markdown("---")
st.write("Created by Asimanshu Samal")
