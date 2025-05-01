# news-sentiment-analyzer
A real-time FastAPI web app for financial news sentiment analysis using FinBERT and BART models.
# News Sentiment Analyzer

A real-time web app that fetches financial news using the Finnhub API and applies AI models for sentiment analysis and summarization. Built with FastAPI and Hugging Face Transformers.

## 🔧 Features
- FastAPI REST endpoint
- FinBERT for financial sentiment analysis
- BART for AI-generated summaries
- Jinja2-based HTML web interface

## 🚀 Technologies
- Python
- FastAPI
- Hugging Face Transformers
- Jinja2
- HTML/CSS

## 🛠 How to Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
