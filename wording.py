"""Optional wording layer. It only rephrases text that the app already wrote.

It never decides, and it never receives raw personal data: only generic template text.
Off by default (config.USE_LLM). Falls back to the original text on any problem.
"""
import os

import config as C
import logger


def rephrase(text: str, goal: str = "Rephrase in simple, friendly English. Keep every number and fact unchanged.") -> str:
    if not C.USE_LLM:
        return text
    try:
        from groq import Groq  # installed only after AP agrees
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return text
        r = Groq(api_key=key).chat.completions.create(
            model=C.GROQ_MODEL, temperature=0.2, max_tokens=300,
            messages=[{"role": "system", "content": goal}, {"role": "user", "content": logger.mask_text(text)}])
        return r.choices[0].message.content.strip() or text
    except Exception:
        return text
