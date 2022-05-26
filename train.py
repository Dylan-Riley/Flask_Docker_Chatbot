import random
import numpy as np
import pickle
import json
# ML and NLP
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential
import nltk
from nltk.stem import WordNetLemmatizer
# Get NLTK packages
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("wordnet")


words = []
labels = []
documents = []
ignore_words = ["?","!"]
# Data is just included this time, no need to go fetch from the github
data_file = open("intents - Copy.json").read()
intents = json.loads(data_file)

# Tokenize words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent["tag"]))

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

# Lemmatize words
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

# Check token and lemmatize
print(len(documents), "documents")
print(len(labels), "labels", labels)
print(len(words), "unique lemmatized words", words)

# Save
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(labels, open("labels.pkl","wb"))

# Build training
training = []
output_empty = [0 for _ in range(len(labels))]
for doc in documents:
    bagOfWords = []
    # Tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        if w in pattern_words:
            # 1 if word match in current pattern
            bagOfWords.append(1)
        else:
            bagOfWords.append(0)
    
    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1

    training.append([bagOfWords, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0]) # patterns
train_y = list(training[:,1]) # intents
print("Training data created!")

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Fit and save model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("Model created!")