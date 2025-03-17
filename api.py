from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# 指定你的本地模型路径

from transformers import AutoModel

model = AutoModel.from_pretrained("YingxiangJEason/bert_optimized")



# 加载 Tokenizer 和 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 设置模型为推理模式
model.eval()


# 定义输入数据格式
class ReviewText(BaseModel):
    text: str


# 预测函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Check the number of classes the model predicts
    num_classes = probabilities.shape[0]

    # If there are 2 classes, assume binary classification (positive/negative)
    if num_classes == 2:
        sentiment_scores = {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
    # If there are 3 classes, use your original logic
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

# 测试推理
print(predict_sentiment("This book is amazing!"))

# 创建 API 端点
@app.post("/predict")
def predict(review: ReviewText):
    result = predict_sentiment(review.text)
    return result

# 运行测试
@app.get("/")
def root():
    return {"message": "API is running"}

import requests

url = "http://127.0.0.1:8000/predict"
data = {"text": "This is an amazing product!"}

response = requests.post(url, json=data)
print(response.json())  # Output: {'text': 'This is an amazing product!', 'prediction': 1}
