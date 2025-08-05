#!/usr/bin/env python3
"""
Utility functions for AGNO Stock Recommendation System
"""

import os
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Union, Dict, List
import pandas as pd
import numpy as np

# Cache directory
CACHE_DIR = "./data/cache"

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/agno.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('agno')
    logger.info("Logging initialized")
    
    return logger

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if value is None or value == '' or pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        if value is None or value == '' or pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency"""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{decimal_places}f}%"

def format_large_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000_000_000:
        return f"{num/1_000_000_000_000:.1f}T"
    elif num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def normalize_score(value: float, min_val: float, max_val: float, 
                   target_min: float = 0.0, target_max: float = 10.0) -> float:
    """Normalize a value to a target range"""
    if max_val == min_val:
        return target_min
    
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)

def calculate_moving_average(data: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if len(data) < window:
        return data
    
    ma = []
    for i in range(len(data)):
        if i < window - 1:
            ma.append(data[i])
        else:
            avg = sum(data[i-window+1:i+1]) / window
            ma.append(avg)
    
    return ma

def calculate_volatility(prices: List[float], periods: int = 30) -> float:
    """Calculate price volatility"""
    if len(prices) < 2:
        return 0.0
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
    
    if len(returns) < periods:
        periods = len(returns)
    
    # Calculate standard deviation of returns
    recent_returns = returns[-periods:]
    if len(recent_returns) < 2:
        return 0.0
    
    mean_return = sum(recent_returns) / len(recent_returns)
    variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
    
    # Annualize volatility
    volatility = (variance ** 0.5) * (252 ** 0.5) * 100
    
    return volatility

def cache_data(key: str, data: Any, expiry_hours: int = 24) -> bool:
    """Cache data to disk"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        cache_item = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'expiry_hours': expiry_hours
        }
        
        cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_item, f)
        
        return True
        
    except Exception as e:
        logging.error(f"Error caching data for key {key}: {e}")
        return False

def load_cached_data(key: str, max_age: timedelta = None) -> Optional[Any]:
    """Load cached data from disk"""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'rb') as f:
            cache_item = pickle.load(f)
        
        # Check expiry
        cached_time = datetime.fromisoformat(cache_item['timestamp'])
        expiry_hours = cache_item.get('expiry_hours', 24)
        
        if max_age:
            expiry_time = cached_time + max_age
        else:
            expiry_time = cached_time + timedelta(hours=expiry_hours)
        
        if datetime.now() > expiry_time:
            # Cache expired
            os.remove(cache_file)
            return None
        
        return cache_item['data']
        
    except Exception as e:
        logging.error(f"Error loading cached data for key {key}: {e}")
        return None

def clear_cache(key: Optional[str] = None) -> bool:
    """Clear cache data"""
    try:
        if key:
            # Clear specific key
            cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            # Clear all cache
            if os.path.exists(CACHE_DIR):
                for file in os.listdir(CACHE_DIR):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(CACHE_DIR, file))
        
        return True
        
    except Exception as e:
        logging.error(f"Error clearing cache: {e}")
        return False

def validate_stock_symbol(symbol: str) -> bool:
    """Validate stock symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.strip().upper()
    
    # Basic validation
    if len(symbol) < 1 or len(symbol) > 5:
        return False
    
    # Must be alphabetic (with possible dot for class shares)
    if not symbol.replace('.', '').isalpha():
        return False
    
    return True

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Basic cleaning
    text = str(text).strip()
    
    # Remove extra whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numbers from text"""
    import re
    
    if not text:
        return []
    
    # Find all numbers (including decimals)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers

def calculate_risk_score(volatility: float, beta: float, debt_to_equity: float, 
                        market_cap: float) -> float:
    """Calculate overall risk score"""
    score = 5.0  # Base risk score
    
    # Volatility component (0-40% volatility mapped to 0-3 points)
    vol_score = min(3.0, (volatility / 40.0) * 3.0)
    score += vol_score
    
    # Beta component (0-2 beta mapped to 0-2 points)
    beta_score = min(2.0, max(0, (beta - 1.0) * 2.0))
    score += beta_score
    
    # Debt component (0-200% D/E mapped to 0-2 points)
    debt_score = min(2.0, (debt_to_equity / 200.0) * 2.0)
    score += debt_score
    
    # Market cap component (smaller = riskier)
    if market_cap < 2_000_000_000:  # Small cap
        score += 1.5
    elif market_cap < 10_000_000_000:  # Mid cap
        score += 0.5
    # Large cap gets no additional risk
    
    return min(10.0, max(0.0, score))

def diversification_score(symbols: List[str], sectors: List[str]) -> float:
    """Calculate diversification score for a portfolio"""
    if not symbols:
        return 0.0
    
    score = 5.0  # Base score
    
    # Number of holdings
    if len(symbols) >= 10:
        score += 2.0
    elif len(symbols) >= 5:
        score += 1.0
    elif len(symbols) < 3:
        score -= 2.0
    
    # Sector diversification
    unique_sectors = len(set(sectors)) if sectors else 1
    if unique_sectors >= 5:
        score += 2.0
    elif unique_sectors >= 3:
        score += 1.0
    elif unique_sectors <= 1:
        score -= 1.0
    
    return min(10.0, max(0.0, score))

def format_recommendation_report(recommendations: List[Dict[str, Any]], 
                               query: str) -> str:
    """Format recommendations into a readable report"""
    if not recommendations:
        return "No recommendations available."
    
    report = f"📊 Stock Recommendations for: '{query}'\n"
    report += "=" * 60 + "\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        symbol = rec.get('symbol', 'N/A')
        company = rec.get('company_name', symbol)
        price = rec.get('current_price', 0)
        score = rec.get('recommendation_score', 0)
        risk = rec.get('risk_level', 'Unknown')
        reasoning = rec.get('reasoning', 'No reasoning provided')
        
        report += f"{i}. {company} ({symbol})\n"
        report += f"   Current Price: {format_currency(price)}\n"
        report += f"   Recommendation Score: {score:.1f}/10\n"
        report += f"   Risk Level: {risk.title()}\n"
        report += f"   Analysis: {reasoning}\n\n"
    
    return report

def get_market_hours() -> Dict[str, Any]:
    """Get current market hours information"""
    now = datetime.now()
    
    # Simple US market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Sunday = 6
    is_market_hours = market_open <= now <= market_close
    
    return {
        'is_market_open': is_weekday and is_market_hours,
        'is_weekday': is_weekday,
        'market_open': market_open.time(),
        'market_close': market_close.time(),
        'current_time': now.time(),
        'next_open': market_open if now < market_open else market_open + timedelta(days=1)
    }

def create_project_structure() -> str:
    """Create the recommended project directory structure"""
    structure = """
    agno/
    ├── main.py                 # Main application entry point
    ├── config.py              # Configuration settings
    ├── data_collector.py      # Data collection utilities
    ├── stock_analyzer.py      # Stock analysis engine
    ├── vector_store.py        # Vector database operations
    ├── utils.py               # Utility functions
    ├── requirements.txt       # Python dependencies
    ├── .env.example          # Environment variables example
    ├── README.md             # Project documentation
    ├── data/                 # Data storage directory
    │   ├── cache/           # Cached data files
    │   └── lancedb/         # Vector database files
    └── logs/                 # Application logs
        └── agno.log         # Main log file
    """
    return structure

class PerformanceTimer:
    """Simple performance timer context manager"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        logging.info(f"⏱️  {self.name} took {duration.total_seconds():.2f} seconds")

# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing AGNO Utilities...")
    
    # Test safe conversions
    print(f"safe_float('123.45'): {safe_float('123.45')}")
    print(f"safe_float('invalid'): {safe_float('invalid', 0.0)}")
    
    # Test formatting
    print(f"format_currency(1234567.89): {format_currency(1234567.89)}")
    print(f"format_large_number(1234567890): {format_large_number(1234567890)}")
    print(f"format_percentage(15.678): {format_percentage(15.678)}")
    
    # Test volatility calculation
    prices = [100, 102, 98, 105, 103, 99, 107, 104]
    vol = calculate_volatility(prices)
    print(f"Volatility for sample prices: {vol:.2f}%")
    
    # Test caching
    cache_data("test_key", {"test": "data"})
    cached = load_cached_data("test_key")
    print(f"Cached data: {cached}")
    
    # Test market hours
    market_info = get_market_hours()
    print(f"Market open: {market_info['is_market_open']}")
    
    print("✅ Utility tests completed!")