# Toy AI Detector
A simple little AI detector I made in a day

## How to use

Clone:  `git clone https://github.com/slothscript/AI-Detector.git`

Install dependancies:
```
pip install nltk
pip install spacy
pip install joblib
```

Get training data:
```
python3 data/gatherHumanData.py
python3 data/gatherAIData.py
```

When gathering AI data I found a value of 500 samples to give around 8000 sentences

Start Flask server: `python3 frontend/server.py`

Go to server at `127.0.0.1:5000`
