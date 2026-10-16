"""Wrap any LLM so every call (prompt and reply, or the error) is recorded. For showing what the
LLM was asked, e.g. in the Streamlit app. It changes nothing about the call itself."""


class RecordingLLM:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []  # [{"system", "user", "reply" | "error"}]

    def complete_json(self, system, user, max_tokens=700):
        record = {"system": system, "user": user}
        self.calls.append(record)
        try:
            reply = self.inner.complete_json(system, user, max_tokens)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        record["reply"] = reply
        return reply
