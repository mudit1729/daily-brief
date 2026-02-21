import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://localhost/daily_brief')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True, 'pool_size': 5}

    # LLM
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5.2')
    LLM_DAILY_TOKEN_BUDGET = int(os.getenv('LLM_DAILY_TOKEN_BUDGET', '100000'))
    LLM_DAILY_BUDGET_USD = float(os.getenv('LLM_DAILY_BUDGET_USD', '1.00'))
    LLM_SECTION_BUDGETS = {
        'general_news_us': 0.20,
        'market': 0.20,
        'ai_news': 0.15,
        'general_news_india': 0.10,
        'general_news_geopolitics': 0.10,
        'science': 0.08,
        'health': 0.07,
        'investment_thesis': 0.10,
    }

    # Embeddings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'openai')

    # Scheduler
    SCHEDULER_ENABLED = os.getenv('SCHEDULER_ENABLED', 'true').lower() == 'true'
    SCHEDULER_API_ENABLED = False

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_ENGINE_OPTIONS = {}  # SQLite doesn't support pool_size
    SCHEDULER_ENABLED = False
    LLM_DAILY_TOKEN_BUDGET = 1000
    LLM_DAILY_BUDGET_USD = 0.10
    OPENAI_API_KEY = 'test-key'
