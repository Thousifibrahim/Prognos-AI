#!/usr/bin/env python3
"""
Vector store module for AGNO Stock Recommendation System
Handles vector database operations using LanceDB
"""

import os
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import json

import lancedb
import numpy as np
import pandas as pd

from config import config
from utils import setup_logging

logger = setup_logging()

class VectorStore:
    """Handles vector database operations for storing and retrieving embeddings"""
    
    def __init__(self):
        self.db_path = config.LANCEDB_PATH
        self.db = None
        self.table_name = "stock_contexts"
        self.table = None
        
    async def initialize(self):
        """Initialize the vector database"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Connect to LanceDB
            self.db = lancedb.connect(self.db_path)
            
            # Initialize or load the table
            await self._initialize_table()
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def _initialize_table(self):
        """Initialize the vector table"""
        try:
            # Check if table exists
            existing_tables = self.db.table_names()
            
            if self.table_name in existing_tables:
                self.table = self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            else:
                # Create new table with schema
                initial_data = self._create_initial_data()
                self.table = self.db.create_table(self.table_name, initial_data)
                logger.info(f"Created new table: {self.table_name}")
                
        except Exception as e:
            logger.error(f"Error initializing table: {e}")
            raise
    
    def _create_initial_data(self) -> List[Dict[str, Any]]:
        """Create initial data for the table schema"""
        # Create some sample financial contexts
        initial_contexts = [
            {
                "id": "sample_1",
                "query": "artificial intelligence stocks",
                "context": "AI and machine learning companies with strong growth potential",
                "symbols": ["NVDA", "GOOGL", "MSFT"],
                "themes": ["AI", "technology", "growth"],
                "timestamp": datetime.now().isoformat(),
                "vector": np.random.rand(config.EMBEDDING_DIMENSION).tolist()
            },
            {
                "id": "sample_2", 
                "query": "dividend income stocks",
                "context": "High-dividend yield stocks for income generation",
                "symbols": ["JNJ", "PG", "KO"],
                "themes": ["dividend", "income", "stability"],
                "timestamp": datetime.now().isoformat(),
                "vector": np.random.rand(config.EMBEDDING_DIMENSION).tolist()
            },
            {
                "id": "sample_3",
                "query": "renewable energy investment",
                "context": "Clean energy and sustainable technology companies",
                "symbols": ["TSLA", "NEE", "ENPH"],
                "themes": ["renewable", "energy", "sustainability"],
                "timestamp": datetime.now().isoformat(),
                "vector": np.random.rand(config.EMBEDDING_DIMENSION).tolist()
            }
        ]
        
        return initial_contexts
    
    async def add_context(self, query: str, context: str, symbols: List[str], 
                         themes: List[str], vector: Optional[List[float]] = None) -> str:
        """Add a new context to the vector store"""
        try:
            # Generate ID
            context_id = self._generate_id(query, context)
            
            # Generate vector if not provided (using simple text-based embedding)
            if vector is None:
                vector = self._generate_simple_embedding(f"{query} {context}")
            
            # Create record
            record = {
                "id": context_id,
                "query": query,
                "context": context,
                "symbols": symbols,
                "themes": themes,
                "timestamp": datetime.now().isoformat(),
                "vector": vector
            }
            
            # Add to table
            self.table.add([record])
            
            logger.info(f"Added context: {context_id}")
            return context_id
            
        except Exception as e:
            logger.error(f"Error adding context: {e}")
            raise
    
    async def search_similar(self, query: str, limit: int = 10, 
                           threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar contexts"""
        try:
            threshold = threshold or config.VECTOR_SIMILARITY_THRESHOLD
            
            # Generate query vector
            query_vector = self._generate_simple_embedding(query)
            
            # Search similar vectors
            results = self.table.search(query_vector).limit(limit).to_pandas()
            
            if results.empty:
                return []
            
            # Convert to list of dictionaries
            similar_contexts = []
            for _, row in results.iterrows():
                # Calculate similarity score (cosine similarity approximation)
                similarity = self._calculate_similarity(query_vector, row['vector'])
                
                if similarity >= threshold:
                    context = {
                        'id': row['id'],
                        'query': row['query'],
                        'context': row['context'],
                        'symbols': row['symbols'] if isinstance(row['symbols'], list) else [],
                        'themes': row['themes'] if isinstance(row['themes'], list) else [],
                        'similarity': similarity,
                        'timestamp': row['timestamp']
                    }
                    similar_contexts.append(context)
            
            # Sort by similarity
            similar_contexts.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similar_contexts)} similar contexts for query: {query}")
            return similar_contexts
            
        except Exception as e:
            logger.error(f"Error searching similar contexts: {e}")
            return []
    
    async def get_context_by_themes(self, themes: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Get contexts by themes"""
        try:
            all_contexts = self.table.to_pandas()
            
            if all_contexts.empty:
                return []
            
            matching_contexts = []
            
            for _, row in all_contexts.iterrows():
                row_themes = row['themes'] if isinstance(row['themes'], list) else []
                
                # Check for theme overlap
                overlap = len(set(themes).intersection(set(row_themes)))
                if overlap > 0:
                    context = {
                        'id': row['id'],
                        'query': row['query'],
                        'context': row['context'],
                        'symbols': row['symbols'] if isinstance(row['symbols'], list) else [],
                        'themes': row_themes,
                        'theme_overlap': overlap,
                        'timestamp': row['timestamp']
                    }
                    matching_contexts.append(context)
            
            # Sort by theme overlap
            matching_contexts.sort(key=lambda x: x['theme_overlap'], reverse=True)
            
            return matching_contexts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting contexts by themes: {e}")
            return []
    
    async def update_context(self, context_id: str, **updates) -> bool:
        """Update an existing context"""
        try:
            # Note: LanceDB doesn't support direct updates, so we'd need to
            # delete and re-add. For now, we'll log this as a limitation.
            logger.warning("Context updates not implemented - LanceDB limitation")
            return False
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return False
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context"""
        try:
            # Note: LanceDB delete functionality
            self.table.delete(f"id = '{context_id}'")
            logger.info(f"Deleted context: {context_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting context: {e}")
            return False
    
    async def get_all_contexts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all contexts"""
        try:
            df = self.table.to_pandas()
            
            if df.empty:
                return []
            
            contexts = []
            for _, row in df.iterrows():
                context = {
                    'id': row['id'],
                    'query': row['query'],
                    'context': row['context'],
                    'symbols': row['symbols'] if isinstance(row['symbols'], list) else [],
                    'themes': row['themes'] if isinstance(row['themes'], list) else [],
                    'timestamp': row['timestamp']
                }
                contexts.append(context)
            
            if limit:
                contexts = contexts[:limit]
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting all contexts: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            df = self.table.to_pandas()
            
            if df.empty:
                return {
                    'total_contexts': 0,
                    'unique_themes': 0,
                    'unique_symbols': 0,
                    'oldest_entry': None,
                    'newest_entry': None
                }
            
            # Extract all themes and symbols
            all_themes = set()
            all_symbols = set()
            
            for _, row in df.iterrows():
                if isinstance(row['themes'], list):
                    all_themes.update(row['themes'])
                if isinstance(row['symbols'], list):
                    all_symbols.update(row['symbols'])
            
            stats = {
                'total_contexts': len(df),
                'unique_themes': len(all_themes),
                'unique_symbols': len(all_symbols),
                'themes': list(all_themes),
                'symbols': list(all_symbols),
                'oldest_entry': df['timestamp'].min(),
                'newest_entry': df['timestamp'].max()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _generate_id(self, query: str, context: str) -> str:
        """Generate unique ID for a context"""
        content = f"{query}:{context}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple text embedding (placeholder for real embedding model)"""
        # This is a very simple approach - in production, use a proper embedding model
        # like sentence-transformers, OpenAI embeddings, or similar
        
        # Convert text to lowercase and split into words
        words = text.lower().split()
        
        # Create a simple hash-based embedding
        vector = np.zeros(config.EMBEDDING_DIMENSION)
        
        for i, word in enumerate(words):
            # Simple hash function to map words to vector positions
            word_hash = hash(word) % config.EMBEDDING_DIMENSION
            vector[word_hash] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            # Cosine similarity
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def add_stock_recommendation_context(self, query: str, recommendations: List[Dict[str, Any]]) -> str:
        """Add a stock recommendation context to the vector store"""
        try:
            # Extract symbols and themes from recommendations
            symbols = [rec['symbol'] for rec in recommendations]
            
            # Extract themes from reasoning
            themes = set()
            context_parts = []
            
            for rec in recommendations:
                reasoning = rec.get('reasoning', '')
                context_parts.append(f"{rec['symbol']}: {reasoning}")
                
                # Simple theme extraction from reasoning
                if 'AI' in reasoning or 'artificial intelligence' in reasoning.lower():
                    themes.add('AI')
                if 'growth' in reasoning.lower():
                    themes.add('growth')
                if 'dividend' in reasoning.lower():
                    themes.add('dividend')
                if 'value' in reasoning.lower():
                    themes.add('value')
                if 'tech' in reasoning.lower() or 'technology' in reasoning.lower():
                    themes.add('technology')
                if 'energy' in reasoning.lower():
                    themes.add('energy')
                if 'healthcare' in reasoning.lower():
                    themes.add('healthcare')
            
            context = "; ".join(context_parts)
            
            context_id = await self.add_context(
                query=query,
                context=context,
                symbols=symbols,
                themes=list(themes)
            )
            
            return context_id
            
        except Exception as e:
            logger.error(f"Error adding stock recommendation context: {e}")
            return ""
    
    async def cleanup_old_contexts(self, days_old: int = 30) -> int:
        """Clean up contexts older than specified days"""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
            
            df = self.table.to_pandas()
            if df.empty:
                return 0
            
            # Convert timestamp strings to datetime
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            
            # Find old contexts
            old_contexts = df[df['timestamp_dt'] < cutoff_date]
            
            deleted_count = 0
            for _, row in old_contexts.iterrows():
                if await self.delete_context(row['id']):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old contexts")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old contexts: {e}")
            return 0

# Utility functions for vector operations
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.linalg.norm(a_np - b_np)

if __name__ == "__main__":
    # Test the vector store
    import asyncio
    
    async def test():
        vector_store = VectorStore()
        await vector_store.initialize()
        
        # Test adding context
        context_id = await vector_store.add_context(
            query="growth stocks with AI exposure",
            context="Technology companies with artificial intelligence capabilities",
            symbols=["NVDA", "GOOGL", "MSFT"],
            themes=["AI", "growth", "technology"]
        )
        
        print(f"Added context: {context_id}")
        
        # Test searching
        similar = await vector_store.search_similar("artificial intelligence investments", limit=3)
        print(f"Found {len(similar)} similar contexts")
        
        for ctx in similar:
            print(f"- {ctx['query']} (similarity: {ctx['similarity']:.3f})")
        
        # Test stats
        stats = await vector_store.get_stats()
        print(f"Vector store stats: {stats}")
    
    asyncio.run(test())