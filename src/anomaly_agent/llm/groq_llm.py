"""A small, safe wrapper around Groq's chat API. It always returns a JSON object (a dict).

Key handling: the key comes from the GROQ_API_KEY environment variable, or from a local `.env`
file in the project root (git-ignored). It is never printed, logged, or put in error messages.

Model: GROQ_MODEL env var / .env, else DEFAULT_MODEL. JSON mode (`response_format json_object`)
works on every Groq model, so we use it and validate the reply ourselves.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile is not available to every account


class LLMError(RuntimeError):
    """The LLM call failed or returned something unusable. Callers fall back to rules."""


class LLMUnavailable(LLMError):
    """No API key (or no SDK): the LLM features cannot run."""


def _read_env_file(path=ENV_FILE):
    values = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def get_api_key(env_path=ENV_FILE):
    return os.environ.get("GROQ_API_KEY") or _read_env_file(env_path).get("GROQ_API_KEY")


def get_model(env_path=ENV_FILE):
    return os.environ.get("GROQ_MODEL") or _read_env_file(env_path).get("GROQ_MODEL") or DEFAULT_MODEL


class GroqLLM:
    def __init__(self, model=None, api_key=None, client=None, temperature=0.0):
        self.model = model or get_model()
        self.temperature = temperature
        if client is None:
            key = api_key or get_api_key()
            if not key:
                raise LLMUnavailable("GROQ_API_KEY is not set (environment variable or .env file)")
            try:
                import groq
            except ImportError as e:
                raise LLMUnavailable("the 'groq' package is not installed") from e
            client = groq.Groq(api_key=key, timeout=30.0, max_retries=2)
        self.client = client

    @staticmethod
    def available():
        return bool(get_api_key())

    def complete_json(self, system, user, max_tokens=700):
        """Ask for a JSON object. Raises LLMError on any failure (never leaks the key)."""
        extra = {}
        if "gpt-oss" in self.model:
            # reasoning models think before answering and the thinking counts toward the token limit:
            # keep it short and leave room, or the JSON answer gets cut off
            extra["reasoning_effort"] = "low"
            max_tokens = max(max_tokens, 2000)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_completion_tokens=max_tokens,
                **extra,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:  # the SDK raises several error classes; all mean "no usable answer"
            raise LLMError(f"Groq request failed ({type(e).__name__}): {str(getattr(e, 'message', e))[:200]}") from e
        try:
            reply = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError("Groq returned text that is not valid JSON") from e
        if not isinstance(reply, dict):
            raise LLMError("Groq returned JSON that is not an object")
        return reply
