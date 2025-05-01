# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from transformers import pipeline
from typing import List, Dict
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load models
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
generator = pipeline("text-generation", model="gpt2")  # Replace QA model with GPT-2 for generative answers

# Store fetched news (global variable for demo purpose)
news_cache = []

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    global news_cache
    # Fetch news
    url = "https://finnhub.io/api/v1/news?category=general&token=cvtskbpr01qjg1363is0cvtskbpr01qjg1363isg"
    response = requests.get(url)
    data = response.json()
    news_cache = []

    for i, item in enumerate(data[:5]):
        title = item["headline"]
        original_summary = item["summary"]
        sentiment = sentiment_analyzer(original_summary)[0]["label"]
        generated_summary = summarizer(original_summary, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

        news_cache.append({
            "id": i,
            "title": title,
            "original_summary": original_summary,
            "generated_summary": generated_summary,
            "sentiment": sentiment
        })

    return templates.TemplateResponse("index.html", {"request": request, "articles": news_cache})

@app.post("/ask-question", response_class=HTMLResponse)
def ask_question(request: Request, article_id: int = Form(...), question: str = Form(...)):
    global news_cache
    context = news_cache[article_id]["original_summary"]
    input_text = f"News: {context}\nQuestion: {question}\nAnswer:"
    answer = generator(input_text, max_length=100, do_sample=True, temperature=0.7)[0]["generated_text"].split("Answer:")[-1].strip()

    return templates.TemplateResponse("answer.html", {
        "request": request,
        "question": question,
        "context": context,
        "answer": answer
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
