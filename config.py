import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me')
    ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://localhost/daily_brief')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True, 'pool_size': 5}

    # LLM
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5.2')
    LLM_DAILY_TOKEN_BUDGET = int(os.getenv('LLM_DAILY_TOKEN_BUDGET', '100000'))
    LLM_DAILY_BUDGET_USD = float(os.getenv('LLM_DAILY_BUDGET_USD', '1.00'))
    LLM_SECTION_BUDGETS = {
        'general_news_us': 0.15,
        'market': 0.15,
        'ai_news': 0.12,
        'general_news_india': 0.08,
        'general_news_geopolitics': 0.08,
        'science': 0.07,
        'health': 0.06,
        'investment_thesis': 0.11,   # includes hedge fund analysis
        'timelines': 0.10,           # auto-update + auto-discover timelines
    }

    # Hedge Fund Analysis
    HEDGE_FUND_TICKERS = os.getenv('HEDGE_FUND_TICKERS', 'AAPL,MSFT,NVDA,GOOGL,AMZN')
    HEDGE_FUND_ANALYSTS = os.getenv('HEDGE_FUND_ANALYSTS', 'technicals,valuation,sentiment,warren_buffett')
    HEDGE_FUND_MODEL_PROVIDER = os.getenv('HEDGE_FUND_MODEL_PROVIDER', 'OpenAI')
    FINANCIAL_DATASETS_API_KEY = os.getenv('FINANCIAL_DATASETS_API_KEY')

    # Embeddings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'openai')

    # Scheduler
    SCHEDULER_ENABLED = os.getenv('SCHEDULER_ENABLED', 'true').lower() == 'true'
    SCHEDULER_API_ENABLED = False
    SOURCE_FAILURE_THRESHOLD = int(os.getenv('SOURCE_FAILURE_THRESHOLD', '3'))
    SOURCE_AUTO_DISABLE_MINUTES = int(os.getenv('SOURCE_AUTO_DISABLE_MINUTES', '180'))
    SOURCE_LATENCY_ALPHA = float(os.getenv('SOURCE_LATENCY_ALPHA', '0.30'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


class TestConfig(Config):
    TESTING = True
    ADMIN_API_KEY = 'test-admin-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_ENGINE_OPTIONS = {}  # SQLite doesn't support pool_size
    SCHEDULER_ENABLED = False
    LLM_DAILY_TOKEN_BUDGET = 1000
    LLM_DAILY_BUDGET_USD = 0.10
    OPENAI_API_KEY = 'test-key'
