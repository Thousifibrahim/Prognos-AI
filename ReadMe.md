

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![AI](https://img.shields.io/badge/AI-powered-orange.svg)
![Vector DB](https://img.shields.io/badge/Vector_DB-LanceDB-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**Intelligent investment insights powered by Groq LLMs and real-time market data.**

# 🚀 AGNO Based — AI-Powered Stock Recommendation System

### 🎯 What is AGNO?
AGNO is a sophisticated financial assistant that bridges the gap between complex market data and natural language. It allows users to query investment ideas in plain English, performing real-time analysis through a RAG (Retrieval-Augmented Generation) pipeline.
#
### ✨ Features

|  **AI-Powered Analysis** |  **Real-Time Market Data** |
| :--- | :--- |
| • Groq API integration for ultra-fast LLM inference. | • Live stock price tracking via `yfinance`. |
| • Natural language query understanding & intent mapping. | • Real-time news scraping via DuckDuckGo. |
| ** Smart Search & Storage** | **⚖️ Risk Assessment** |
| • Vector similarity search using **LanceDB**. | • Dynamic risk scoring based on volatility. |
| • Persistent context storage for personalized advice. | • Portfolio alignment with user-defined goals. |

#
### 🛠️ Installation & Setup

### **Prerequisites**
* **Python 3.8+**
* **Groq API Key** ([Get it here](https://console.groq.com/))

### **Quick Install**
```bash
# 1. Clone & Enter Directory
git clone [https://github.com/your-username/agno.git](https://github.com/your-username/agno.git) && cd agno

# 2. Install Core Dependencies
pip install groq lancedb yfinance duckduckgo-search pandas python-dotenv tqdm

# 3. Configure Environment
echo "GROQ_API_KEY=your_key_here" > .env

```

#
### Quick Start  :

### **1. Initialize Database**

```bash
python setup.py

```

### **2. Launch AGNO**

python main.py

#
### **Example Queries**

* *"Find high-growth AI stocks with moderate risk."*
* *"Suggest dividend-paying healthcare stocks for a retirement portfolio."*
* *"Analyze NVIDIA's current market sentiment vs. competitors."*

#
### 🏗️ Project Structure

```
agno/
├── src/
│   ├── main.py            # Application Entry Point
│   ├── stock_analyzer.py  # LLM & Logic Layer
│   ├── data_collector.py  # Finance API Integration
│   └── vector_store.py    # LanceDB Vector Operations
├── data/
│   ├── cache/             # Temporary API Responses
│   └── lancedb/           # Vectorized Financial Data
├── config.py              # System Hyperparameters
└── .env                   # Environment Secrets (DO NOT COMMIT)
```

#
### 🔧 Advanced Configuration

You can fine-tune the recommendation engine in `config.py`:

```python
# Example Configuration Settings
MAX_RECOMMENDATIONS = 5      # Number of stocks to suggest
CACHE_EXPIRY = 3600          # Refresh market data every hour
RISK_TOLERANCE_LEVEL = "MED" # Global default risk setting 
```
#

**Made with ❤️ by Thousif ibrahim**

[⬆ Back to Top](https://www.google.com/search?q=%23-agno---ai-powered-stock-recommendation-system)
