"""A fake LLM for tests and offline demos: you decide what it answers.

It has the same method as GroqLLM (`complete_json`), so anything that takes an LLM takes this.
If the responder returns an Exception, it is raised (to test failure handling).
"""


class ScriptedLLM:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []  # (system, user) of every call, so tests can inspect the prompts

    def complete_json(self, system, user, max_tokens=700):
        self.calls.append((system, user))
        reply = self.responder(system, user)
        if isinstance(reply, Exception):
            raise reply
        return reply
