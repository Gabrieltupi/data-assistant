from unittest.mock import MagicMock, patch

import pytest

from src.llm.factory import get_chat_model


def test_unknown_provider_raises() -> None:
    with patch("src.llm.factory.settings") as mock_settings:
        mock_settings.llm_provider = "grok"
        with pytest.raises(ValueError, match="anthropic"):
            get_chat_model()


def test_claude_alias_uses_anthropic() -> None:
    fake = MagicMock()
    with (
        patch("src.llm.factory.settings") as mock_settings,
        patch("langchain_anthropic.ChatAnthropic", return_value=fake) as ctor,
    ):
        mock_settings.llm_provider = "claude"
        mock_settings.anthropic_model = "claude-sonnet-4-5"
        mock_settings.anthropic_api_key = "sk-ant-test"
        assert get_chat_model() is fake
        ctor.assert_called_once_with(
            model="claude-sonnet-4-5",
            api_key="sk-ant-test",
            temperature=0,
        )
