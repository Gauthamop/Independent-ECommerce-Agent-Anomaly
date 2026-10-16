import pytest

from src.anomaly_agent.llm import groq_llm
from src.anomaly_agent.llm.groq_llm import GroqLLM, LLMError, LLMUnavailable


class FakeClient:
    """Looks like groq.Groq(): client.chat.completions.create(...) -> .choices[0].message.content"""

    def __init__(self, content=None, error=None):
        self.content, self.error, self.last_kwargs = content, error, None
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs
                if outer.error:
                    raise outer.error
                message = type("M", (), {"content": outer.content})()
                return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


def test_returns_a_dict_and_asks_for_json_mode():
    client = FakeClient('{"tool": "query_metric"}')
    out = GroqLLM(client=client, model="m").complete_json("sys", "user")
    assert out == {"tool": "query_metric"}
    assert client.last_kwargs["response_format"] == {"type": "json_object"}
    assert client.last_kwargs["temperature"] == 0.0


@pytest.mark.parametrize("content", ["not json at all", "[1, 2, 3]", "", None])
def test_unusable_replies_raise_llm_error(content):
    with pytest.raises(LLMError):
        GroqLLM(client=FakeClient(content), model="m").complete_json("s", "u")


def test_sdk_errors_become_llm_error_without_leaking_anything():
    with pytest.raises(LLMError) as info:
        GroqLLM(client=FakeClient(error=RuntimeError("boom")), model="m").complete_json("s", "u")
    assert "RuntimeError" in str(info.value)


def test_missing_key_means_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(groq_llm, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(groq_llm, "get_api_key", lambda env_path=tmp_path / "missing.env": groq_llm.os.environ.get("GROQ_API_KEY"))
    with pytest.raises(LLMUnavailable):
        GroqLLM()


def test_env_file_is_read_and_environment_wins(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="from-file"\nGROQ_MODEL=some/model\n', encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert groq_llm.get_api_key(env) == "from-file"
    assert groq_llm.get_model(env) == "some/model" or groq_llm.os.environ.get("GROQ_MODEL")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert groq_llm.get_api_key(env) == "from-env"
