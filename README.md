# Flask_Docker_Chatbot

Quick demo of a intents-based chatbot running in a flask environment

Bot is trained in train.py off of intents.json and then saves its work in the .pkl and itself in the .h5 files. Bot is then ran in app.py, which just running this can then be connected to at localhost:5000

### KNOWN ISSUES:
* Trying to build the Dockerfile from my machine's frozen requirements.txt causes errors
  * with h5py (it tries to build from raw)
  * with tensorflow (version numbers cannot be found)

### SEE ALSO:
More impressive chatbot from my group: 
https://github.com/alasali1/shoestorechatbot
