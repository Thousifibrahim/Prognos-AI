# 🚀 AGNO - AI-Powered Stock Recommendation System

AGNO is an intelligent stock recommendation system that uses natural language processing, financial data analysis, and vector similarity search to provide personalized stock recommendations based on user queries.

## ✨ Features

- **Natural Language Queries**: Ask for stocks in plain English
- **AI-Powered Analysis**: Uses Groq API for intelligent query understanding
- **Real-time Data**: Fetches live stock data using yfinance
- **News Integration**: Incorporates market news using DuckDuckGo search
- **Vector Search**: Stores and retrieves similar investment contexts using LanceDB
- **Risk Assessment**: Comprehensive risk scoring for each recommendation
- **Personalized Recommendations**: Tailored suggestions based on risk tolerance and investment goals

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (get one at [console.groq.com](https://console.groq.com/))

### Setup Steps

1. **Clone or create the project directory:**
```bash
mkdir agno
cd agno
```

2. **Install required packages:**
```bash
pip install groq duckduckgo-search pypdf lancedb pandas yfinance tantivy numpy scipy requests aiohttp python-dotenv tqdm
```

3. **Set up environment variables:**
```bash
# Create a .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

Or export directly:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

4. **Create project structure:**
```bash
mkdir -p data/cache data/lancedb logs
```

## 🚦 Quick Start

1. **Save all the provided Python files** in your `agno` directory:
   - `main.py`
   - `config.py`
   - `data_collector.py`
   - `stock_analyzer.py`
   - `vector_store.py`
   - `utils.py`

2. **Run the system:**
```bash
python  setup.py # This will initialize the database and download necessary data
python main.py
```

3. **Start asking for stock recommendations:**
```
💬 What kind of stocks are you looking for?
> I want AI stocks with high growth potential

💬 What kind of stocks are you looking for?
> Show me dividend stocks for retirement income

💬 What kind of stocks are you looking for?
> I'm looking for value stocks in the healthcare sector
```

## 📄 Documentation
For detailed documentation on how to use AGNO, refer to the [[AGNO Documentation](https://docs.agno.com/)]


Happy coding! If you have any questions or need help, feel free to open an issue on the [GitHub repository]