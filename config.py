#!/usr/bin/env python3
"""
Configuration settings for AGNO Stock Recommendation System
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    """Configuration class for the stock recommendation system"""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    
    # Database settings
    LANCEDB_PATH: str = './data/lancedb'
    
    # Stock data settings
    DEFAULT_PERIOD: str = '1y'  # Default period for stock data
    DEFAULT_INTERVAL: str = '1d'  # Default interval for stock data
    
    # Recommendation settings
    MAX_RECOMMENDATIONS: int = 10
    MIN_MARKET_CAP: int = 1_000_000_000  # 1 billion minimum market cap
    
    # News and search settings
    MAX_NEWS_ARTICLES: int = 10
    NEWS_SEARCH_DAYS: int = 7
    
    # Vector store settings
    EMBEDDING_DIMENSION: int = 384
    VECTOR_SIMILARITY_THRESHOLD: float = 0.7
    
    # Risk scoring weights
    RISK_WEIGHTS: dict = None
    
    # Popular stock lists by category
    POPULAR_STOCKS: dict = None
    
    def __post_init__(self):
        """Initialize complex configurations after object creation"""
        
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Risk scoring weights
        if self.RISK_WEIGHTS is None:
            self.RISK_WEIGHTS = {
                'volatility': 0.3,
                'beta': 0.2,
                'debt_to_equity': 0.2,
                'pe_ratio': 0.15,
                'market_cap': 0.15
            }
        
        # Popular stocks by category
        if self.POPULAR_STOCKS is None:
            self.POPULAR_STOCKS = {
                'tech': [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 
                    'CRM', 'ADBE', 'INTC', 'ORCL', 'IBM', 'NFLX'
                ],
                'finance': [
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW'
                ],
                'healthcare': [
                    'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'DHR', 'BMY'
                ],
                'energy': [
                    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE'
                ],
                'consumer': [
                    'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'COST'
                ],
                'industrial': [
                    'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT'
                ],
                'materials': [
                    'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'SHW'
                ],
                'utilities': [
                    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE'
                ],
                'real_estate': [
                    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB'
                ]
            }
    
    def get_stocks_by_sector(self, sector: str) -> List[str]:
        """Get stock symbols by sector"""
        sector_lower = sector.lower()
        
        # Map common sector names to our categories
        sector_mapping = {
            'technology': 'tech',
            'financial': 'finance',
            'healthcare': 'healthcare',
            'energy': 'energy',
            'consumer discretionary': 'consumer',
            'consumer staples': 'consumer',
            'industrials': 'industrial',
            'materials': 'materials',
            'utilities': 'utilities',
            'real estate': 'real_estate',
            'communication services': 'tech'
        }
        
        mapped_sector = sector_mapping.get(sector_lower, sector_lower)
        return self.POPULAR_STOCKS.get(mapped_sector, [])
    
    def get_all_stocks(self) -> List[str]:
        """Get all tracked stock symbols"""
        all_stocks = []
        for stocks in self.POPULAR_STOCKS.values():
            all_stocks.extend(stocks)
        return list(set(all_stocks))  # Remove duplicates
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        if not self.GROQ_API_KEY:
            print("❌ GROQ_API_KEY is required")
            return False
            
        if self.MAX_RECOMMENDATIONS <= 0:
            print("❌ MAX_RECOMMENDATIONS must be greater than 0")
            return False
            
        if self.MIN_MARKET_CAP <= 0:
            print("❌ MIN_MARKET_CAP must be greater than 0")
            return False
            
        return True

# Create global config instance
config = Config()

# Environment setup instructions
SETUP_INSTRUCTIONS = """
🔧 AGNO Setup Instructions:

1. Install required packages:
   pip install groq duckduckgo-search pypdf lancedb pandas yfinance tantivy

2. Set environment variables:
   export GROQ_API_KEY="your_groq_api_key_here"
   
   Or create a .env file:
   GROQ_API_KEY=your_groq_api_key_here

3. Get your Groq API key from: https://console.groq.com/

4. Run the system:
   python main.py

📁 Project Structure:
agno/
├── main.py              # Main entry point
├── config.py            # Configuration settings
├── data_collector.py    # Data collection utilities
├── stock_analyzer.py    # Stock analysis logic
├── vector_store.py      # Vector database operations
├── utils.py             # Utility functions
├── requirements.txt     # Package dependencies
└── data/               # Data storage directory
    └── lancedb/        # Vector database files
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    
    # Validate configuration
    if config.validate_config():
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration validation failed!")