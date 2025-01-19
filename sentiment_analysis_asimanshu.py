from transformers import pipeline
import numpy
text=input("Enter the text you want to analyze here:")
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
result = classifier(text)
mood=""
if 0<=result[0]['score']<0.25:
    mood="Unsure, but "
elif 0.25<=result[0]['score']<0.5:
    mood="Somewhat "
elif 0.5<=result[0]['score']<0.7:
    mood="Moderately "
elif 0.7<=result[0]['score']<0.85:
    mood="Likely "
elif 0.85<=result[0]['score']<0.95:
    mood="Strongly "
else:
    mood="Definitely"
if result[0]['label']=="LABEL_2":
    mood= mood+"Positive"
elif result[0]['label']=="LABEL_1":
    mood= mood+"Neutral"
elif result[0]['label']=="LABEL_0":
    mood= mood+"Negative"
else:
    print('Something went wrong, please try again later.')
print("Our servers have recieved the text '",text,"' which recieves a sentiment of",mood,".")
further=input("Would you like to know more about our sentiment measuring algorithm? Type yes if so.")
if further.lower()=="yes":
    print("Here are the advanced metrics of our analysis:\n",result,"\n Here, the LABEL signifies the emotion and score signifies the strength of that emotion.\n LABEL_0 is negative, LABEL_1 is neutral and LABEL_2 is positive.\n The scores range from 0 to 1, is 0 being completely unsure and 1 being sure.")
else:
    pass
