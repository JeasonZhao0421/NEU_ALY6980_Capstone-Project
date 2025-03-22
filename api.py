from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import uvicorn


#  Creating a FastAPI Instance
app = FastAPI()

# Read Hugging Face Token
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Defining model paths
model_path = "YingxiangJEason/bert_optimized"

# Loading Tokenizer and BERT Models
tokenizer = BertTokenizer.from_pretrained(model_path, token=hf_token)
model = BertForSequenceClassification.from_pretrained(model_path, token=hf_token)

# Setting the model to reasoning mode
model.eval()

# Define input data format
class ReviewText(BaseModel):
    text: str

# predictive function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Check the number of classes the model predicts
    num_classes = probabilities.shape[0]

    # Binary classification (positive/negative)
    if num_classes == 2:
        sentiment_scores = {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
    # Three-class classification
    elif num_classes == 3:
        sentiment_scores = {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]),
            "positive": float(probabilities[2])
        }
    else:
        raise ValueError(f"Unexpected number of classes: {num_classes}")

    predicted_label = max(sentiment_scores, key=sentiment_scores.get)
    return {"predicted_sentiment": predicted_label, "probabilities": sentiment_scores}


# Creating API Endpoints
@app.post("/predict")
def predict(review: ReviewText):
    result = predict_sentiment(review.text)
    return result

# API Health Check
@app.get("/")
def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render may provide different ports
    uvicorn.run(app, host="0.0.0.0", port=port)

