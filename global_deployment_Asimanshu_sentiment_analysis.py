import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis App")

text = st.text_area("Enter text to analyze:", height=150)

try:
    @st.cache_resource
    def load_mod():
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    classifier = load_mod()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            try:
                result = classifier(text)
                predicted_label = result[0]['label']  # Correct way to access the label (string)
                score = result[0]['score']

                # Map the label string to your output
                if predicted_label == "LABEL_2":
                    sentiment = "Positive"
                elif predicted_label == "LABEL_1":
                    sentiment = "Neutral"
                elif predicted_label == "LABEL_0":
                    sentiment = "Negative"
                else:
                    sentiment = "Unknown"  # Handle unexpected labels
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

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Sentiment", value=mood)
                    st.metric(label=" Score", value=f"{score:.3f}")
                    st.progress(score)
                    if sentiment == "Positive":
                        st.success("Positive Sentiment Detected! :thumbsup:")
                    elif sentiment == "Negative":
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
st.write("Created by Asimanshu Samal")
