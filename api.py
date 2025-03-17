from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# 创建 FastAPI 实例
app = FastAPI()

# 读取 Hugging Face Token
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# **定义模型路径**
model_path = "YingxiangJEason/bert_optimized"

# **加载 Tokenizer 和 BERT 模型**
tokenizer = BertTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
model = BertForSequenceClassification.from_pretrained(model_path, use_auth_token=hf_token)

# **设置模型为推理模式**
model.eval()

# **定义输入数据格式**
class ReviewText(BaseModel):
    text: str

# **预测函数**
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

# **创建 API 端点**
@app.post("/predict")
def predict(review: ReviewText):
    result = predict_sentiment(review.text)
    return result

# **API 健康检查**
@app.get("/")
def root():
    return {"message": "API is running"}
