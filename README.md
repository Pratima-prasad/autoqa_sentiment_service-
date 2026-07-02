# AutoQA Sentiment Service

A FastAPI-based REST API for detecting **Rude** and **Neutral** customer utterances in Hinglish (Hindi + English) conversations using a fine-tuned **XLM-RoBERTa** transformer model.

The API supports:
- Single text prediction
- Batch prediction
- CSV file prediction
- Automatic model loading
- Fast inference using PyTorch and Hugging Face Transformers


#  Features

- ✅ Hinglish Sentiment Classification
- ✅ FastAPI REST API
- ✅ Batch Prediction
- ✅ CSV Upload Support
- ✅ Probability Score
- ✅ Configurable Threshold
- ✅ GPU/CPU Support
- ✅ Easy Deployment on AWS EC2

---

# Project Structure

```
autoqa_sentiment_service/
│
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── rude_detector.py
├── requirements.txt
├── .gitignore
├── README.md
└── model_link_to_download.txt
```

> **Note:** The model (`model.safetensors`) is not included in this repository because GitHub does not allow files larger than 100 MB.

---

# Model

The trained model can be downloaded from Google Drive.

**Model Download Link**

https://drive.google.com/file/d/1bUs0nsRMYrv4b0wf0vxnRvsI_ThPL03d/view?usp=sharing

After downloading:

Copy

```
model.safetensors
```

into the project folder.

# Installation

Clone the repository

```bash
git clone https://github.com/Pratima-prasad/autoqa_sentiment_service-.git

cd autoqa_sentiment_service-
```

Install dependencies

```bash
pip install -r requirements.txt
```

#  Running the Server

Start the FastAPI server

```bash
python rude_detector.py
```

or

```bash
uvicorn rude_detector:app --host 0.0.0.0 --port 8000
```

Server starts at

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

ReDoc Documentation

```
http://localhost:8000/redoc
```


# API Endpoints

## Health Check

```
GET /health
```

Returns server status and model information.


## Predict Single Text

```
POST /predict
```

Example

```json
{
    "text":"Aap bilkul bekaar service de rahe ho."
}
```

Response

```json
{
    "prediction":"rude",
    "rude_prob":0.97
}
```

## Batch Prediction

```
POST /predict_batch
```

Example

```json
{
    "texts":[
        "Thank you",
        "Tum pagal ho"
    ]
}
```
## CSV Prediction

```
POST /predict_csv
```

Upload a CSV file containing a column named

```
transcript
```

The API returns

- Original CSV
- Prediction
- Rude Probability

## Reload Model

```
POST /reload
```

Reloads the model without restarting the server.

# Tech Stack

- Python
- FastAPI
- PyTorch
- HuggingFace Transformers
- XLM-RoBERTa
- Pandas
- Uvicorn

# Requirements

Install all packages using

```bash
pip install -r requirements.txt
```

# Model Information

Model:

```
XLM-RoBERTa Base
```

Framework

```
PyTorch
```

Classification

```
Binary Classification
```

Labels

```
Neutral
Rude
```


# Author

**Krish Gupta**

GitHub

https://github.com/Krish229200
