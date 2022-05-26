import random
import numpy as np
import pickle
import json
import os
# Flask
from flask import Flask, render_template, request
# ML and NLP
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("wordnet")

# Init lemmatizer
lemmatizer = WordNetLemmatizer()

model=load_model("chatbot_model.h5")
intents = json.loads(open("intents - Copy.json").read())
words = pickle.load(open("words.pkl","rb"))
labels = pickle.load(open("labels.pkl","rb"))

app = Flask(__name__)

# Landing page
@app.route("/")
def home():
    return render_template("index.html")

# User sends message
@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    #print(type(msg))
    ints = predict_label(str(msg), model)
    res = getResponse(ints, intents)
    return res

# Clean user message
def cleanUpSentence(sentence):
    # Tokenize
    sentenceWords = nltk.word_tokenize(sentence)
    # Lemmatize
    sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
    return sentenceWords

def bagOfWords(sentence, words, show_details=True):
    sentenceWords = cleanUpSentence(sentence)

    bag = [0 for _ in range(len(words))]
    for s in sentenceWords:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Predict response label
def predict_label(sentence, model):
    p = bagOfWords(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse = True)
    returnList = []
    for r in results:
        returnList.append({"intent":labels[r[0]], "probability":str(r[1])})
    return returnList

# Select response
def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result=random.choice(i["responses"])
            break
    return result

#if __name__ == "__main__":
#    app.run()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)