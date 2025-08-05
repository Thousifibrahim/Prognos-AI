#!/usr/bin/env python3
"""
Setup script for AGNO Stock Recommendation System
Run this script to set up the project environment
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'data',
        'data/cache',
        'data/lancedb',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env.example file"""
    env_content = """# AGNO Configuration
# Get your Groq API key from: https://console.groq.com/

GROQ_API_KEY=your_groq_api_key_here

# Optional: Override default settings
# LANCEDB_PATH=./data/lancedb
# MAX_RECOMMENDATIONS=10
# MIN_MARKET_CAP=1000000000
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env.example file")
    
    if not os.path.exists('.env'):
        print("📝 Please copy .env.example to .env and add your Groq API key")

def install_dependencies():
    """Install required Python packages"""
    packages = [
        'groq>=0.4.0',
        'duckduckgo-search>=3.9.0', 
        'pypdf>=3.17.0',
        'lancedb>=0.3.0',
        'pandas>=2.0.0',
        'yfinance>=0.2.0',
        'tantivy>=0.20.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'requests>=2.31.0',
        'aiohttp>=3.8.0',
        'python-dotenv>=1.0.0',
        'tqdm>=4.65.0'
    ]
    
    print("📦 Installing required packages...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Data and Logs
data/cache/*
data/lancedb/*
logs/*
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Keep directory structure but ignore contents
!data/cache/.gitkeep
!data/lancedb/.gitkeep
!logs/.gitkeep
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    # Create .gitkeep files
    for path in ['data/cache/.gitkeep', 'data/lancedb/.gitkeep', 'logs/.gitkeep']:
        Path(path).touch()
    
    print("✅ Created .gitignore file")

def validate_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def check_groq_api_key():
    """Check if Groq API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key or api_key == 'your_groq_api_key_here':
        print("⚠️  Groq API key not configured")
        print("   1. Get your API key from: https://console.groq.com/")
        print("   2. Copy .env.example to .env")
        print("   3. Add your API key to the .env file")
        return False
    
    print("✅ Groq API key configured")
    return True

def run_basic_test():
    """Run a basic test to ensure everything is working"""
    try:
        # Test imports
        import groq
        import pandas
        import yfinance
        import lancedb
        import numpy
        from duckduckgo_search import DDGS
        
        print("✅ All required packages imported successfully")
        
        # Test basic functionality
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'longName' in info:
            print(f"✅ yfinance test successful: {info['longName']}")
        else:
            print("⚠️  yfinance test failed - may be rate limited")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Test warning: {e}")
        return True  # Non-critical error

def main():
    """Main setup function"""
    print("🚀 AGNO Stock Recommendation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not validate_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    
    # Install dependencies
    print("\n📦 Installing Dependencies...")
    if not install_dependencies():
        print("❌ Failed to install some dependencies")
        print("   You may need to install them manually")
    
    # Run basic tests
    print("\n🧪 Running Basic Tests...")
    run_basic_test()
    
    # Check API key configuration
    print("\n🔑 Checking API Configuration...")
    api_configured = check_groq_api_key()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    if api_configured:
        print("✅ Your system is ready to run!")
        print("   Run: python main.py")
    else:
        print("⚠️  Setup complete, but API key needed:")
        print("   1. Get API key from: https://console.groq.com/")
        print("   2. Copy .env.example to .env")
        print("   3. Add your API key to .env")
        print("   4. Run: python main.py")
    
    print("\n📚 Next Steps:")
    print("   • Read README.md for detailed usage instructions")
    print("   • Check the examples in the documentation")
    print("   • Start with simple queries like 'AI stocks' or 'dividend stocks'")
    
    print("\n🆘 If you encounter issues:")
    print("   • Check the troubleshooting section in README.md")
    print("   • Ensure your internet connection is stable")
    print("   • Verify all packages are installed correctly")

if __name__ == "__main__":
    main()