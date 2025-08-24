# Medium AI Content Curator API

> A powerful API that intelligently curates Medium articles and provides AI-powered content recommendations.

## 🚀 What It Does

- Scrapes articles from Medium using web scraping
- Uses AI/ML to score and recommend articles
- Provides a REST API for getting personalized suggestions
- Tracks user preferences and interactions

## 🛠️ Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API server:**
   ```bash
   python api_server.py
   ```

3. **Test the endpoints:**
   ```bash
   python test_endpoints.py
   ```

## 📡 Main Endpoints

- `GET /health` - Health check
- `POST /api/suggestions` - Get AI recommendations
- `GET /api/trending` - Trending articles
- `POST /api/search` - Search articles
- `GET /api/topics` - Available topics

## 🧪 Testing

The API includes comprehensive tests that verify all endpoints work correctly.

## 🚧 Status

**This is a development version that will be tested in production.**
