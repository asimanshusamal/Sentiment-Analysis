import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis App")

text = st.text_area("Enter text to analyze:", height=150)

@st.cache_data  # Correctly using st.cache_data
def loadmod():
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment") #basically the default model was really mediocre, so I found a model I liked which was trained using real social media data
    except Exception as e: #error handling 
        st.error(f"Error loading the model: {e}")
        st.stop()

classifier = loadmod() #I learnt these terms especially Streamlit mostly from the internet, but the base was in class

if st.button("Analyze Text"):
    if (not text.strip()) == True: #Removing spaces and whitespace character
        st.warning("Please enter some text to analyze.")
    else:
        st.spinner('Analyzing...')
        try:
            #This is the part I wrote in Jupyter Notebook, before even knowing what streamlit was
            result = classifier(text)
            predicted_label = result[0]['label']
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
            if predicted_label == "LABEL_2": 
                mood= mood+" Positive"
            elif predicted_label == "LABEL_1":
                mood= mood+" Neutral"
            elif predicted_label == "LABEL_0":
                mood= mood+" Negative"
            else:
                sentiment = "Unknown" #basic error handling
                st.error("Unexpected label returned by the model.")
            st.write(f"Sentiment: {mood}")
            st.write(f"Score: {score:.2f}")
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
