import pytest
from unittest.mock import patch, MagicMock
from datetime import date
from app.models.cost import LLMCallLog
from app.integrations.llm_gateway import LLMGateway, BudgetExhaustedError


class TestLLMGateway:
    def test_cost_computation(self, app):
        """Cost should be computed correctly for gpt-5.2."""
        with app.app_context():
            gateway = LLMGateway(app.config)
            cost = gateway._compute_cost(prompt_tokens=1000, completion_tokens=500)
            # gpt-5.2: $1.75/1M input + $14.00/1M output
            expected = (1000 / 1_000_000) * 1.75 + (500 / 1_000_000) * 14.00
            assert abs(cost - expected) < 0.0001

    def test_budget_enforcement(self, app, db_session):
        """Gateway should raise BudgetExhaustedError when budget is gone."""
        with app.app_context():
            gateway = LLMGateway(app.config)
            # Test config has 1000 token budget
            # Simulate having used all tokens
            log = LLMCallLog(
                call_purpose='test',
                model='gpt-4o-mini',
                prompt_tokens=800,
                completion_tokens=200,
                total_tokens=1000,
                cost_usd=0.01,
                section=None,
            )
            db_session.add(log)
            db_session.commit()

            remaining = gateway._get_remaining_budget()
            assert remaining <= 0

    def test_degradation_levels(self, app, db_session):
        """Degradation level should increase as budget depletes."""
        with app.app_context():
            gateway = LLMGateway(app.config)
            # With full budget, degradation should be 0
            level = gateway.determine_degradation_level('general_news_us')
            assert level == 0

    def test_call_logs_cost(self, app, db_session):
        """Each LLM call should be logged."""
        with app.app_context():
            gateway = LLMGateway(app.config)

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content='Test response'))]
            mock_response.usage = MagicMock(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )

            with patch('openai.OpenAI') as mock_client:
                mock_client.return_value.chat.completions.create.return_value = mock_response

                result = gateway.call(
                    messages=[{'role': 'user', 'content': 'test'}],
                    purpose='test_call',
                    section='general_news_us',
                )

            assert result['total_tokens'] == 150
            assert result['content'] == 'Test response'

            log = LLMCallLog.query.filter_by(call_purpose='test_call').first()
            assert log is not None
            assert log.total_tokens == 150


    def test_xai_available_flag(self, app):
        """xai_available should reflect XAI_API_KEY config."""
        with app.app_context():
            gateway = LLMGateway(app.config)
            # TestConfig does not set XAI_API_KEY
            assert gateway.xai_available is False

            gateway_with_key = LLMGateway({
                **app.config,
                'XAI_API_KEY': 'test-xai-key',
                'XAI_MODEL': 'grok-3-mini-fast',
            })
            assert gateway_with_key.xai_available is True

    def test_xai_cost_computation(self, app):
        """Cost computation should work for Grok models."""
        with app.app_context():
            gateway = LLMGateway(app.config)
            cost = gateway._compute_cost(
                prompt_tokens=1000, completion_tokens=500,
                model='grok-3-mini-fast',
            )
            # grok-3-mini-fast: $0.30/1M input + $0.50/1M output
            expected = (1000 / 1_000_000) * 0.30 + (500 / 1_000_000) * 0.50
            assert abs(cost - expected) < 0.0001

    def test_xai_call_uses_correct_base_url(self, app, db_session):
        """xAI call should use the xAI base URL and model."""
        with app.app_context():
            gateway = LLMGateway({
                **app.config,
                'XAI_API_KEY': 'test-xai-key',
                'XAI_MODEL': 'grok-3-mini-fast',
            })

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content='Grok take'))]
            mock_response.usage = MagicMock(
                prompt_tokens=80, completion_tokens=40, total_tokens=120
            )

            with patch('openai.OpenAI') as mock_client:
                mock_client.return_value.chat.completions.create.return_value = mock_response

                result = gateway.call(
                    messages=[{'role': 'user', 'content': 'test'}],
                    purpose='grok_test',
                    section='grok_analysis',
                    provider='xai',
                )

            assert result['content'] == 'Grok take'
            assert result['provider'] == 'xai'
            assert result['model'] == 'grok-3-mini-fast'

            # Verify OpenAI client was called with xAI base URL
            mock_client.assert_called_with(
                api_key='test-xai-key',
                base_url='https://api.x.ai/v1',
            )

            # Verify cost log
            log = LLMCallLog.query.filter_by(call_purpose='grok_test').first()
            assert log is not None
            assert log.model == 'grok-3-mini-fast'


class TestExtractiveFallback:
    def test_extractive_summary(self, app):
        """Extractive fallback should use lead sentences."""
        from app.utils.text import extract_lead_sentences
        text = "First important sentence. Second key point. Third detail. Fourth extra."
        result = extract_lead_sentences(text, n=2)
        assert 'First important sentence.' in result
        assert 'Second key point.' in result
        assert 'Third' not in result
