# 🚀 AGNO - AI-Powered Stock Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![AI](https://img.shields.io/badge/AI-powered-orange.svg)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)

*Intelligent stock recommendations powered by AI and real-time market data*

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [📖 Documentation](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 What is AGNO?

AGNO is an intelligent stock recommendation system that transforms natural language queries into personalized investment suggestions. Using advanced AI, real-time market data, and comprehensive risk analysis, AGNO helps investors make informed decisions.

<div align="center">
  
**💬 Ask in Plain English** → **🤖 AI Processing** → **📊 Data Analysis** → **📈 Smart Recommendations**

</div>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 **AI-Powered Analysis**
- Natural language query processing
- Intelligent context understanding
- Groq API integration

### 📊 **Real-Time Market Data**
- Live stock prices via yfinance
- Current market trends
- Historical performance analysis

</td>
<td width="50%">

### 🔍 **Smart Search & Storage**
- Vector similarity search with LanceDB
- News integration via DuckDuckGo
- Persistent investment context storage

### ⚖️ **Risk Assessment**
- Comprehensive risk scoring
- Personalized risk tolerance matching
- Investment goal alignment

</td>
</tr>
</table>

## 🛠️ Installation

### Prerequisites

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-API_Key-FF6B35?style=for-the-badge)](https://console.groq.com/)

</div>

### 📥 Setup Steps

<details>
<summary><b>🔧 Method 1: Quick Setup (Recommended)</b></summary>

```bash
# 1. Create project directory
mkdir agno && cd agno

# 2. Install dependencies
pip install groq duckduckgo-search pypdf lancedb pandas yfinance tantivy numpy scipy requests aiohttp python-dotenv tqdm

# 3. Set up environment
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Create directory structure
mkdir -p data/cache data/lancedb logs
```

</details>

<details>
<summary><b>🐍 Method 2: Virtual Environment (Recommended for Production)</b></summary>

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Clone/create project
mkdir agno && cd agno

# 3. Install dependencies
pip install -r requirements.txt  # or use the pip install command above

# 4. Configure environment
cp .env.example .env  # Edit with your API key
```

</details>

### 🔑 API Key Setup

1. **Get your Groq API key:**
   - Visit [console.groq.com](https://console.groq.com/)
   - Sign up and create a new API key
   - Copy your key

2. **Set environment variable:**

   **Windows:**
   ```cmd
   set GROQ_API_KEY=your_actual_api_key_here
   ```
   
   **Linux/Mac:**
   ```bash
   export GROQ_API_KEY="your_actual_api_key_here"
   ```

## 🚦 Quick Start

### 🎬 Interactive Demo

```bash
# Initialize the system
python setup.py

# Start AGNO
python main.py
```

### 💡 Example Queries

<div align="center">

| Query Type | Example |
|------------|---------|
| 🤖 **AI/Tech Stocks** | *"I want AI stocks with high growth potential"* |
| 💰 **Dividend Stocks** | *"Show me dividend stocks for retirement income"* |
| 🏥 **Sector-Specific** | *"I'm looking for value stocks in healthcare"* |
| 🔍 **Risk-Based** | *"Conservative stocks for long-term investment"* |

</div>

### 📱 Sample Interaction

```
🚀 AGNO Stock Recommendation System
💬 What kind of stocks are you looking for?

> I want AI stocks with high growth potential

🔍 Analyzing your request...
📊 Found 5 AI stocks matching your criteria:

1. 🎯 NVIDIA (NVDA) - Score: 92/100
   💹 Price: $450.23 (+2.4%)
   ⚖️ Risk: Medium-High
   📈 Growth Potential: Excellent

2. 🎯 Microsoft (MSFT) - Score: 89/100
   💹 Price: $378.91 (+1.2%)
   ⚖️ Risk: Medium
   📈 Growth Potential: Very Good

[... more recommendations ...]

💡 Want more details on any stock? Just ask!
```

## 📁 Project Structure

```
agno/
├── 📄 main.py              # Main application entry point
├── ⚙️ config.py            # Configuration management
├── 📊 data_collector.py    # Stock data collection
├── 🧠 stock_analyzer.py    # AI-powered analysis
├── 🗄️ vector_store.py      # Vector database operations
├── 🛠️ utils.py             # Utility functions
├── 📋 setup.py             # Database initialization
├── 📁 data/
│   ├── 💾 cache/           # Cached data
│   └── 🗃️ lancedb/         # Vector database
└── 📁 logs/                # Application logs
```

## 🔧 Configuration

<details>
<summary><b>Environment Variables</b></summary>

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
CACHE_DURATION=3600          # Cache duration in seconds
MAX_RECOMMENDATIONS=10       # Maximum recommendations per query
LOG_LEVEL=INFO              # Logging level
```

</details>

<details>
<summary><b>Advanced Settings</b></summary>

Edit `config.py` to customize:
- API endpoints
- Cache settings
- Risk calculation parameters
- Vector similarity thresholds

</details>

## 📖 Documentation

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=for-the-badge&logo=gitbook)](https://docs.agno.com/)
[![API Reference](https://img.shields.io/badge/API-reference-blue?style=for-the-badge&logo=swagger)](https://api.agno.com/docs)

</div>

- **📚 [User Guide](https://docs.agno.com/user-guide)** - Complete usage instructions
- **🔧 [API Reference](https://docs.agno.com/api)** - Technical documentation
- **💡 [Examples](https://docs.agno.com/examples)** - Use case examples
- **❓ [FAQ](https://docs.agno.com/faq)** - Frequently asked questions

## 🤝 Contributing

We welcome contributions! Here's how you can help:

<div align="center">

[![Issues](https://img.shields.io/badge/report-issues-red?style=for-the-badge&logo=github)](https://github.com/your-repo/agno/issues)
[![Pull Requests](https://img.shields.io/badge/submit-PRs-green?style=for-the-badge&logo=github)](https://github.com/your-repo/agno/pulls)
[![Discussions](https://img.shields.io/badge/join-discussions-blue?style=for-the-badge&logo=github)](https://github.com/your-repo/agno/discussions)

</div>

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. 💾 Commit your changes
4. 📤 Push to the branch
5. 🔀 Open a Pull Request

## 📊 Statistics

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/your-repo/agno?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-repo/agno?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-repo/agno)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-repo/agno)

</div>

## 📄 License

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

</div>

## 🙏 Acknowledgments

- **Groq** for powerful AI processing
- **yfinance** for financial data
- **LanceDB** for vector storage
- **DuckDuckGo** for news integration

---

<div align="center">

**Made with ❤️ by Smd-Designs**

[⬆ Back to Top](#-agno---ai-powered-stock-recommendation-system)

</div>