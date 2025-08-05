#!/usr/bin/env python3
"""
Stock analysis engine for AGNO Stock Recommendation System
Handles stock scoring, ranking, and recommendation logic
"""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import config
from data_collector import DataCollector
from utils import safe_float, normalize_score

logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Handles stock analysis and recommendation generation"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        
    async def analyze_stocks(self, query_analysis: Dict[str, Any], 
                           news_context: List[Dict[str, Any]], 
                           similar_contexts: List[Dict[str, Any]], 
                           num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Main function to analyze stocks and generate recommendations"""
        
        # Get candidate stocks based on query analysis
        candidate_stocks = await self._get_candidate_stocks(query_analysis)
        
        if not candidate_stocks:
            logger.warning("No candidate stocks found")
            return []
        
        # Fetch stock data for candidates
        stock_data = await self.data_collector.update_stock_data(candidate_stocks)
        
        if not stock_data:
            logger.warning("No stock data retrieved")
            return []
        
        # Score each stock
        scored_stocks = []
        for symbol, data in stock_data.items():
            try:
                score_data = await self._score_stock(data, query_analysis, news_context)
                if score_data:
                    scored_stocks.append(score_data)
            except Exception as e:
                logger.error(f"Error scoring stock {symbol}: {e}")
                continue
        
        # Sort by total score
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top recommendations
        return scored_stocks[:num_recommendations]
    
    async def _get_candidate_stocks(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Get candidate stocks based on query analysis"""
        candidates = set()
        
        # Add explicitly mentioned companies
        if 'companies_mentioned' in query_analysis:
            candidates.update(query_analysis['companies_mentioned'])
        
        # Add stocks from mentioned sectors
        if 'sectors_mentioned' in query_analysis:
            for sector in query_analysis['sectors_mentioned']:
                sector_stocks = config.get_stocks_by_sector(sector)
                candidates.update(sector_stocks)
        
        # Add stocks based on themes
        if 'key_themes' in query_analysis:
            theme_stocks = await self._get_stocks_by_themes(query_analysis['key_themes'])
            candidates.update(theme_stocks)
        
        # If no specific candidates, use popular stocks filtered by criteria
        if not candidates:
            candidates = self._filter_stocks_by_criteria(config.get_all_stocks(), query_analysis)
        
        return list(candidates)
    
    async def _get_stocks_by_themes(self, themes: List[str]) -> List[str]:
        """Get stocks related to specific themes"""
        theme_mapping = {
            'AI': ['NVDA', 'AMD', 'GOOGL', 'MSFT', 'META', 'CRM', 'ADBE'],
            'artificial intelligence': ['NVDA', 'AMD', 'GOOGL', 'MSFT', 'META', 'CRM', 'ADBE'],
            'renewable energy': ['TSLA', 'NEE', 'ENPH', 'SEDG', 'FSLR', 'BEP'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'DHR'],
            'fintech': ['SQ', 'PYPL', 'V', 'MA', 'ADYEN', 'SHOP'],
            'cloud computing': ['AMZN', 'MSFT', 'GOOGL', 'CRM', 'SNOW', 'NET'],
            'electric vehicles': ['TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID'],
            'cybersecurity': ['CRWD', 'ZS', 'OKTA', 'PANW', 'FTNT', 'S'],
            'gaming': ['NVDA', 'AMD', 'ATVI', 'EA', 'TTWO', 'RBLX'],
            'streaming': ['NFLX', 'DIS', 'ROKU', 'SPOT', 'PARA'],
            'social media': ['META', 'SNAP', 'TWTR', 'PINS', 'RBLX'],
            'e-commerce': ['AMZN', 'SHOP', 'ETSY', 'EBAY', 'MELI'],
            'biotechnology': ['MRNA', 'BNTX', 'GILD', 'BIIB', 'REGN']
        }
        
        theme_stocks = set()
        for theme in themes:
            theme_lower = theme.lower()
            if theme_lower in theme_mapping:
                theme_stocks.update(theme_mapping[theme_lower])
        
        return list(theme_stocks)
    
    def _filter_stocks_by_criteria(self, stocks: List[str], query_analysis: Dict[str, Any]) -> List[str]:
        """Filter stocks by market cap and other criteria"""
        # For now, return a reasonable subset
        # In production, you'd filter based on actual market cap data
        
        market_cap_pref = query_analysis.get('market_cap_preference', 'any')
        
        if market_cap_pref == 'large':
            # Return large cap stocks (simplified)
            large_cap = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B']
            return [s for s in large_cap if s in stocks]
        elif market_cap_pref == 'small':
            # Return smaller, more growth-oriented names
            return stocks[-20:]  # Last 20 as proxy for smaller names
        else:
            return stocks[:30]  # Top 30 as default
    
    async def _score_stock(self, stock_data: Dict[str, Any], 
                          query_analysis: Dict[str, Any], 
                          news_context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Score a single stock based on multiple factors"""
        
        try:
            # Calculate individual scores
            fundamental_score = self._calculate_fundamental_score(stock_data)
            technical_score = self._calculate_technical_score(stock_data)
            risk_score = self._calculate_risk_score(stock_data)
            sentiment_score = self._calculate_sentiment_score(stock_data, news_context)
            alignment_score = self._calculate_query_alignment_score(stock_data, query_analysis)
            
            # Weight the scores based on query analysis
            weights = self._get_scoring_weights(query_analysis)
            
            total_score = (
                fundamental_score * weights['fundamental'] +
                technical_score * weights['technical'] +
                (10 - risk_score) * weights['risk'] +  # Invert risk score
                sentiment_score * weights['sentiment'] +
                alignment_score * weights['alignment']
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(stock_data, {
                'fundamental': fundamental_score,
                'technical': technical_score,
                'risk': risk_score,
                'sentiment': sentiment_score,
                'alignment': alignment_score
            })
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score, stock_data)
            
            # Calculate price target (simplified)
            price_target = self._calculate_price_target(stock_data)
            
            return {
                'symbol': stock_data['symbol'],
                'score': min(total_score, 10.0),  # Cap at 10
                'reasoning': reasoning,
                'risk_level': risk_level,
                'price_target': price_target,
                'scores': {
                    'fundamental': fundamental_score,
                    'technical': technical_score,
                    'risk': risk_score,
                    'sentiment': sentiment_score,
                    'alignment': alignment_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error scoring stock {stock_data.get('symbol', 'Unknown')}: {e}")
            return None
    
    def _calculate_fundamental_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score (0-10)"""
        score = 5.0  # Base score
        
        # P/E ratio scoring
        pe_ratio = stock_data.get('pe_ratio')
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                score += 1.5
            elif 5 <= pe_ratio < 10 or 20 < pe_ratio <= 25:
                score += 1.0
            elif pe_ratio > 30:
                score -= 1.0
        
        # PEG ratio scoring
        peg_ratio = stock_data.get('peg_ratio')
        if peg_ratio:
            if peg_ratio < 1.0:
                score += 1.5
            elif peg_ratio <= 1.5:
                score += 1.0
            elif peg_ratio > 2.0:
                score -= 0.5
        
        # ROE scoring
        roe = stock_data.get('roe')
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                score += 1.5
            elif roe_pct > 15:
                score += 1.0
            elif roe_pct < 5:
                score -= 1.0
        
        # Profit margin scoring
        profit_margin = stock_data.get('profit_margin')
        if profit_margin:
            margin_pct = profit_margin * 100
            if margin_pct > 20:
                score += 1.0
            elif margin_pct > 10:
                score += 0.5
            elif margin_pct < 5:
                score -= 0.5
        
        # Debt to equity scoring
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        if debt_to_equity < 30:
            score += 0.5
        elif debt_to_equity > 100:
            score -= 1.0
        
        return max(0, min(10, score))
    
    def _calculate_technical_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (0-10)"""
        score = 5.0  # Base score
        
        current_price = stock_data.get('current_price', 0)
        sma_20 = stock_data.get('sma_20', current_price)
        sma_50 = stock_data.get('sma_50', current_price)
        sma_200 = stock_data.get('sma_200', current_price)
        
        # Moving average trend scoring
        if current_price > sma_20 > sma_50 > sma_200:
            score += 2.0  # Strong uptrend
        elif current_price > sma_20 > sma_50:
            score += 1.5  # Moderate uptrend
        elif current_price > sma_200:
            score += 1.0  # Above long-term trend
        elif current_price < sma_200:
            score -= 1.0  # Below long-term trend
        
        # Price momentum scoring
        price_change_30d = stock_data.get('price_change_30d', 0)
        if price_change_30d > 10:
            score += 1.5
        elif price_change_30d > 5:
            score += 1.0
        elif price_change_30d < -10:
            score -= 1.5
        elif price_change_30d < -5:
            score -= 1.0
        
        # Volume analysis
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 0.5  # Above average volume
        elif volume_ratio < 0.5:
            score -= 0.5  # Below average volume
        
        # 52-week position
        high_52w = stock_data.get('high_52w', current_price)
        low_52w = stock_data.get('low_52w', current_price)
        if high_52w > low_52w:
            position_52w = (current_price - low_52w) / (high_52w - low_52w)
            if position_52w > 0.8:
                score += 1.0  # Near 52-week high
            elif position_52w < 0.2:
                score -= 0.5  # Near 52-week low
        
        return max(0, min(10, score))
    
    def _calculate_risk_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate risk score (0-10, higher = more risky)"""
        risk_score = 5.0  # Base risk
        
        # Volatility risk
        volatility = stock_data.get('volatility_30d', 20)
        if volatility > 40:
            risk_score += 2.0
        elif volatility > 25:
            risk_score += 1.0
        elif volatility < 15:
            risk_score -= 1.0
        
        # Beta risk
        beta = stock_data.get('beta', 1.0)
        if beta > 1.5:
            risk_score += 1.5
        elif beta > 1.2:
            risk_score += 0.5
        elif beta < 0.8:
            risk_score -= 0.5
        
        # Market cap risk (smaller = riskier)
        market_cap = stock_data.get('market_cap', 0)
        if market_cap < 2_000_000_000:  # < 2B
            risk_score += 2.0
        elif market_cap < 10_000_000_000:  # < 10B
            risk_score += 1.0
        elif market_cap > 100_000_000_000:  # > 100B
            risk_score -= 1.0
        
        # Debt risk
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        if debt_to_equity > 100:
            risk_score += 1.5
        elif debt_to_equity > 50:
            risk_score += 0.5
        elif debt_to_equity < 20:
            risk_score -= 0.5
        
        return max(0, min(10, risk_score))
    
    def _calculate_sentiment_score(self, stock_data: Dict[str, Any], 
                                  news_context: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score based on news and analyst recommendations"""
        score = 5.0  # Neutral base
        
        # Analyst recommendation scoring
        recommendation = stock_data.get('recommendation', 'hold').lower()
        rec_scores = {
            'strong_buy': 9.0,
            'buy': 7.5,
            'hold': 5.0,
            'sell': 2.5,
            'strong_sell': 1.0
        }
        
        if recommendation in rec_scores:
            score = rec_scores[recommendation]
        
        # News sentiment (simplified - count positive vs negative keywords)
        if news_context:
            positive_keywords = ['growth', 'profit', 'beat', 'strong', 'bullish', 'upgrade', 'buy']
            negative_keywords = ['loss', 'decline', 'weak', 'bearish', 'downgrade', 'sell', 'risk']
            
            positive_count = 0
            negative_count = 0
            
            for article in news_context:
                text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
                
                for keyword in positive_keywords:
                    positive_count += text.count(keyword)
                
                for keyword in negative_keywords:
                    negative_count += text.count(keyword)
            
            if positive_count > negative_count:
                score += min(2.0, (positive_count - negative_count) * 0.2)
            elif negative_count > positive_count:
                score -= min(2.0, (negative_count - positive_count) * 0.2)
        
        return max(0, min(10, score))
    
    def _calculate_query_alignment_score(self, stock_data: Dict[str, Any], 
                                       query_analysis: Dict[str, Any]) -> float:
        """Calculate how well the stock aligns with user query"""
        score = 5.0  # Base alignment
        
        # Risk tolerance alignment
        risk_tolerance = query_analysis.get('risk_tolerance', 'medium')
        stock_risk = self._determine_risk_level(self._calculate_risk_score(stock_data), stock_data)
        
        risk_alignment = {
            ('low', 'low'): 2.0,
            ('low', 'medium'): 0.5,
            ('low', 'high'): -2.0,
            ('medium', 'low'): 1.0,
            ('medium', 'medium'): 2.0,
            ('medium', 'high'): 0.5,
            ('high', 'low'): -1.0,
            ('high', 'medium'): 1.0,
            ('high', 'high'): 2.0
        }
        
        alignment_key = (risk_tolerance, stock_risk)
        if alignment_key in risk_alignment:
            score += risk_alignment[alignment_key]
        
        # Investment goal alignment
        investment_goal = query_analysis.get('investment_goal', 'growth')
        dividend_yield = stock_data.get('dividend_yield', 0)
        pe_ratio = stock_data.get('pe_ratio', 15)
        
        if investment_goal == 'income' and dividend_yield > 3:
            score += 2.0
        elif investment_goal == 'growth' and pe_ratio and pe_ratio > 20:
            score += 1.5
        elif investment_goal == 'value' and pe_ratio and pe_ratio < 15:
            score += 1.5
        
        return max(0, min(10, score))
    
    def _get_scoring_weights(self, query_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Get scoring weights based on query analysis"""
        base_weights = {
            'fundamental': 0.3,
            'technical': 0.2,
            'risk': 0.2,
            'sentiment': 0.15,
            'alignment': 0.15
        }
        
        # Adjust weights based on investment goal
        investment_goal = query_analysis.get('investment_goal', 'growth')
        
        if investment_goal == 'value':
            base_weights['fundamental'] = 0.4
            base_weights['technical'] = 0.15
        elif investment_goal == 'growth':
            base_weights['technical'] = 0.25
            base_weights['sentiment'] = 0.2
        elif investment_goal == 'income':
            base_weights['fundamental'] = 0.4
            base_weights['risk'] = 0.25
        
        return base_weights
    
    def _generate_reasoning(self, stock_data: Dict[str, Any], scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the recommendation"""
        symbol = stock_data['symbol']
        company = stock_data.get('company_name', symbol)
        
        reasons = []
        
        # Fundamental reasons
        if scores['fundamental'] > 7:
            pe_ratio = stock_data.get('pe_ratio')
            roe = stock_data.get('roe')
            if pe_ratio and pe_ratio < 20:
                reasons.append(f"attractive valuation with P/E of {pe_ratio:.1f}")
            if roe and roe > 0.15:
                reasons.append(f"strong profitability with ROE of {roe*100:.1f}%")
        
        # Technical reasons
        if scores['technical'] > 7:
            price_change = stock_data.get('price_change_30d', 0)
            if price_change > 5:
                reasons.append(f"positive momentum with {price_change:.1f}% gain in 30 days")
        
        # Risk assessment
        risk_level = self._determine_risk_level(scores['risk'], stock_data)
        if risk_level == 'low':
            reasons.append("low-risk profile suitable for conservative investors")
        elif risk_level == 'high':
            reasons.append("higher risk but potential for greater returns")
        
        # Sector strength
        sector = stock_data.get('sector', 'Unknown')
        if sector != 'Unknown':
            reasons.append(f"operates in the {sector} sector")
        
        # Analyst sentiment
        recommendation = stock_data.get('recommendation', 'hold')
        if recommendation in ['buy', 'strong_buy']:
            reasons.append("positive analyst sentiment")
        
        if not reasons:
            reasons.append("balanced risk-reward profile")
        
        return f"{company} shows promise due to " + ", ".join(reasons[:3]) + "."
    
    def _determine_risk_level(self, risk_score: float, stock_data: Dict[str, Any]) -> str:
        """Determine risk level category"""
        if risk_score < 4:
            return 'low'
        elif risk_score < 7:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_price_target(self, stock_data: Dict[str, Any]) -> Optional[float]:
        """Calculate a simple price target"""
        current_price = stock_data.get('current_price')
        analyst_target = stock_data.get('analyst_target')
        
        if analyst_target:
            return analyst_target
        
        if current_price:
            # Simple price target based on technical analysis
            sma_20 = stock_data.get('sma_20', current_price)
            price_change_30d = stock_data.get('price_change_30d', 0)
            
            # Basic momentum-based target
            if price_change_30d > 0:
                target = current_price * (1 + min(price_change_30d / 100 * 0.5, 0.2))
            else:
                target = max(current_price * 1.05, sma_20)
            
            return target
        
        return None

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test():
        analyzer = StockAnalyzer()
        
        query_analysis = {
            'investment_goal': 'growth',
            'risk_tolerance': 'medium',
            'time_horizon': 'long',
            'key_themes': ['AI']
        }
        
        recommendations = await analyzer.analyze_stocks(query_analysis, [], [], 3)
        
        print("Test Recommendations:")
        for rec in recommendations:
            print(f"- {rec['symbol']}: {rec['score']:.1f}/10")
            print(f"  {rec['reasoning']}")
    
    asyncio.run(test())