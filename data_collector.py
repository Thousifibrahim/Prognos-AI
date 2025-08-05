#!/usr/bin/env python3
"""
Data collection module for AGNO Stock Recommendation System
Handles stock data, news, and market information gathering
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import yfinance as yf
from duckduckgo_search import DDGS
import numpy as np

from config import config
from utils import safe_float, cache_data, load_cached_data

logger = logging.getLogger(__name__)

class DataCollector:
    """Handles all data collection operations"""
    
    def __init__(self):
        self.cache_duration = timedelta(hours=1)  # Cache data for 1 hour
        
    async def get_stock_data(self, symbol: str, period: str = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive stock data for a symbol"""
        period = period or config.DEFAULT_PERIOD
        
        try:
            # Check cache first
            cache_key = f"stock_data_{symbol}_{period}"
            cached_data = load_cached_data(cache_key, self.cache_duration)
            if cached_data:
                return cached_data
            
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            price_change_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            price_change_30d = ((current_price - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30]) * 100 if len(hist) >= 30 else 0
            
            # Volatility (30-day)
            volatility = hist['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
            
            # Volume analysis
            avg_volume = hist['Volume'].rolling(30).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Support and resistance levels
            high_52w = hist['High'].rolling(252).max().iloc[-1] if len(hist) >= 252 else hist['High'].max()
            low_52w = hist['Low'].rolling(252).min().iloc[-1] if len(hist) >= 252 else hist['Low'].min()
            
            # Moving averages
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else current_price
            
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': current_price,
                'price_change_1d': price_change_1d,
                'price_change_30d': price_change_30d,
                'volatility_30d': volatility,
                'volume_ratio': volume_ratio,
                'beta': safe_float(info.get('beta', 1.0)),
                'pe_ratio': safe_float(info.get('trailingPE')),
                'forward_pe': safe_float(info.get('forwardPE')),
                'peg_ratio': safe_float(info.get('pegRatio')),
                'price_to_book': safe_float(info.get('priceToBook')),
                'debt_to_equity': safe_float(info.get('debtToEquity', 0)),
                'roe': safe_float(info.get('returnOnEquity')),
                'profit_margin': safe_float(info.get('profitMargins')),
                'dividend_yield': safe_float(info.get('dividendYield', 0)) * 100,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'analyst_target': safe_float(info.get('targetMeanPrice')),
                'recommendation': info.get('recommendationKey', 'hold'),
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the data
            cache_data(cache_key, stock_data)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def update_stock_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Update data for multiple stocks concurrently"""
        logger.info(f"Updating data for {len(symbols)} stocks...")
        
        # Use asyncio.gather to fetch data concurrently
        tasks = [self.get_stock_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                continue
            if result:
                stock_data[symbol] = result
        
        logger.info(f"Successfully updated data for {len(stock_data)} stocks")
        return stock_data
    
    async def get_market_news(self, query: str, max_articles: int = None) -> List[Dict[str, Any]]:
        """Get relevant market news using DuckDuckGo search"""
        max_articles = max_articles or config.MAX_NEWS_ARTICLES
        
        try:
            # Enhance query for financial news
            enhanced_query = f"{query} stock market news financial"
            
            ddgs = DDGS()
            news_results = []
            
            # Search for news
            news_search = ddgs.news(
                keywords=enhanced_query,
                region='us-en',
                safesearch='off',
                timelimit='w',  # Past week
                max_results=max_articles
            )
            
            for article in news_search:
                news_item = {
                    'title': article.get('title', ''),
                    'snippet': article.get('body', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'date': article.get('date', ''),
                    'relevance_score': self._calculate_news_relevance(article, query)
                }
                news_results.append(news_item)
            
            # Sort by relevance score
            news_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Found {len(news_results)} relevant news articles")
            return news_results[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _calculate_news_relevance(self, article: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for news article"""
        title = article.get('title', '').lower()
        snippet = article.get('body', '').lower()
        query_lower = query.lower()
        
        # Simple keyword matching scoring
        score = 0.0
        query_words = query_lower.split()
        
        for word in query_words:
            if word in title:
                score += 2.0  # Title matches are more important
            if word in snippet:
                score += 1.0
        
        # Boost score for financial keywords
        financial_keywords = ['stock', 'market', 'earnings', 'revenue', 'profit', 'investment']
        for keyword in financial_keywords:
            if keyword in title or keyword in snippet:
                score += 0.5
        
        return score
    
    async def get_sector_performance(self) -> Dict[str, float]:
        """Get sector performance data"""
        try:
            # Major sector ETFs
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC'
            }
            
            sector_performance = {}
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period='1mo')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        month_ago_price = hist['Close'].iloc[0]
                        performance = ((current_price - month_ago_price) / month_ago_price) * 100
                        sector_performance[sector] = performance
                except Exception as e:
                    logger.warning(f"Could not fetch performance for {sector}: {e}")
                    sector_performance[sector] = 0.0
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment indicators"""
        try:
            sentiment_data = {}
            
            # VIX (Fear index)
            try:
                vix = yf.Ticker('^VIX')
                vix_hist = vix.history(period='5d')
                if not vix_hist.empty:
                    current_vix = vix_hist['Close'].iloc[-1]
                    sentiment_data['vix'] = current_vix
                    sentiment_data['fear_level'] = 'high' if current_vix > 25 else 'medium' if current_vix > 15 else 'low'
            except:
                sentiment_data['vix'] = None
                sentiment_data['fear_level'] = 'unknown'
            
            # Market indices performance
            indices = {'^GSPC': 'sp500', '^DJI': 'dow', '^IXIC': 'nasdaq'}
            
            for symbol, name in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change = ((current - prev) / prev) * 100
                        sentiment_data[f'{name}_change'] = change
                except:
                    sentiment_data[f'{name}_change'] = 0.0
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return {}
    
    async def search_similar_companies(self, company_query: str, limit: int = 10) -> List[str]:
        """Search for similar companies based on query"""
        try:
            # Use DuckDuckGo to search for similar companies
            ddgs = DDGS()
            search_query = f"{company_query} similar companies stocks competitors"
            
            results = ddgs.text(search_query, region='us-en', safesearch='off', max_results=limit)
            
            # Extract potential stock symbols from results
            # This is a simplified approach - in production, you'd want more sophisticated NER
            symbols = []
            
            for result in results:
                text = f"{result.get('title', '')} {result.get('body', '')}"
                # Look for patterns like "AAPL", "MSFT" etc.
                import re
                potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
                symbols.extend(potential_symbols)
            
            # Filter to valid symbols (basic validation)
            valid_symbols = []
            for symbol in symbols:
                if len(symbol) <= 5 and symbol.isalpha():
                    valid_symbols.append(symbol)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for symbol in valid_symbols:
                if symbol not in seen:
                    seen.add(symbol)
                    unique_symbols.append(symbol)
            
            return unique_symbols[:limit]
            
        except Exception as e:
            logger.error(f"Error searching for similar companies: {e}")
            return []

if __name__ == "__main__":
    # Test the data collector
    async def test():
        collector = DataCollector()
        
        # Test stock data
        aapl_data = await collector.get_stock_data('AAPL')
        print("AAPL Data:")
        print(json.dumps(aapl_data, indent=2, default=str))
        
        # Test news search
        news = await collector.get_market_news('artificial intelligence stocks')
        print(f"\nFound {len(news)} news articles")
        for article in news[:3]:
            print(f"- {article['title']}")
    
    asyncio.run(test())