#!/usr/bin/env python3
"""
AGNO - AI-Powered Stock Recommendation System
Main entry point for the stock recommendation system
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from groq import Groq
import pandas as pd
from duckduckgo_search import DDGS
import yfinance as yf
import lancedb
import numpy as np

from config import Config
from data_collector import DataCollector
from stock_analyzer import StockAnalyzer
from vector_store import VectorStore
from utils import setup_logging, format_currency

logger = setup_logging()

class StockRecommendationSystem:
    def __init__(self):
        self.config = Config()
        self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
        self.data_collector = DataCollector()
        self.stock_analyzer = StockAnalyzer()
        self.vector_store = VectorStore()
        
    async def initialize(self):
        """Initialize the system and load data"""
        logger.info("Initializing Stock Recommendation System...")
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Load popular stocks data
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
            'AMD', 'CRM', 'PYPL', 'ADBE', 'INTC', 'ORCL', 'IBM', 'UBER'
        ]
        
        logger.info(f"Loading data for {len(popular_stocks)} stocks...")
        await self.data_collector.update_stock_data(popular_stocks)
        
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query using Groq API to extract intent and parameters"""
        prompt = f"""
        Analyze this investment query and extract key information:
        Query: "{user_query}"
        
        Extract and return in JSON format:
        {{
            "investment_goal": "growth/income/value/speculative",
            "risk_tolerance": "low/medium/high", 
            "time_horizon": "short/medium/long",
            "sectors_mentioned": ["list of sectors if any"],
            "companies_mentioned": ["list of companies if any"],
            "market_cap_preference": "small/mid/large/any",
            "budget_range": "amount if mentioned",
            "key_themes": ["AI", "renewable energy", "healthcare", etc.]
        }}
        
        Be concise and only include fields where you have confidence from the query.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.1
            )
            
            import json
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query with Groq: {e}")
            return {
                "investment_goal": "growth",
                "risk_tolerance": "medium",
                "time_horizon": "medium"
            }
    
    async def get_recommendations(self, user_query: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get stock recommendations based on user query"""
        logger.info(f"Processing recommendation request: {user_query}")
        
        # Analyze the query
        query_analysis = self.analyze_query(user_query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Search for relevant news and context
        news_context = await self.data_collector.get_market_news(user_query)
        
        # Find similar queries/contexts in vector store
        similar_contexts = await self.vector_store.search_similar(user_query, limit=10)
        
        # Get stock recommendations based on analysis
        recommendations = await self.stock_analyzer.analyze_stocks(
            query_analysis, news_context, similar_contexts, num_recommendations
        )
        
        # Format recommendations with current data
        formatted_recommendations = []
        for rec in recommendations:
            try:
                stock_data = yf.Ticker(rec['symbol']).info
                current_price = yf.Ticker(rec['symbol']).history(period='1d')['Close'].iloc[-1]
                
                formatted_rec = {
                    'symbol': rec['symbol'],
                    'company_name': stock_data.get('longName', rec['symbol']),
                    'current_price': current_price,
                    'sector': stock_data.get('sector', 'Unknown'),
                    'market_cap': stock_data.get('marketCap', 0),
                    'recommendation_score': rec['score'],
                    'reasoning': rec['reasoning'],
                    'risk_level': rec['risk_level'],
                    'price_target': rec.get('price_target'),
                    'news_sentiment': rec.get('news_sentiment', 'neutral')
                }
                formatted_recommendations.append(formatted_rec)
                
            except Exception as e:
                logger.error(f"Error formatting recommendation for {rec['symbol']}: {e}")
                continue
        
        return formatted_recommendations
    
    def generate_report(self, recommendations: List[Dict[str, Any]], user_query: str) -> str:
        """Generate a detailed investment report"""
        
        report_prompt = f"""
        Create a professional investment report based on these stock recommendations for the query: "{user_query}"
        
        Recommendations:
        {recommendations}
        
        Structure the report with:
        1. Executive Summary
        2. Market Context
        3. Individual Stock Analysis
        4. Risk Assessment
        5. Conclusion and Next Steps
        
        Make it professional but accessible, with clear reasoning for each recommendation.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": report_prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating detailed report. Please try again."

async def main():
    """Main function to run the stock recommendation system"""
    system = StockRecommendationSystem()
    
    try:
        await system.initialize()
        
        print("🚀 AGNO Stock Recommendation System Ready!")
        print("="*50)
        
        while True:
            user_query = input("\n💬 What kind of stocks are you looking for? (or 'quit' to exit): ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using AGNO!")
                break
            
            if not user_query:
                continue
            
            print("\n🔍 Analyzing your request...")
            recommendations = await system.get_recommendations(user_query)
            
            if not recommendations:
                print("❌ No recommendations found. Please try a different query.")
                continue
            
            print(f"\n📊 Here are {len(recommendations)} stock recommendations:")
            print("="*60)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['company_name']} ({rec['symbol']})")
                print(f"   💰 Current Price: ${rec['current_price']:.2f}")
                print(f"   🏢 Sector: {rec['sector']}")
                print(f"   📈 Score: {rec['recommendation_score']:.1f}/10")
                print(f"   ⚠️  Risk Level: {rec['risk_level']}")
                print(f"   💡 Reasoning: {rec['reasoning']}")
                
                if rec.get('price_target'):
                    print(f"   🎯 Price Target: ${rec['price_target']:.2f}")
            
            # Ask if user wants detailed report
            detailed = input("\n📋 Would you like a detailed investment report? (y/n): ").strip().lower()
            if detailed == 'y':
                print("\n📄 Generating detailed report...")
                report = system.generate_report(recommendations, user_query)
                print("\n" + "="*60)
                print("DETAILED INVESTMENT REPORT")
                print("="*60)
                print(report)
    
    except KeyboardInterrupt:
        print("\n👋 Thanks for using AGNO!")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())