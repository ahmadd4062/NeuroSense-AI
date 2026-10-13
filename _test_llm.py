import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Neurosense.settings")
import django
django.setup()

from recommendations_app.services import get_openrouter_client, RECOMMENDATION_MODEL

client = get_openrouter_client()

r = client.chat.completions.create(
    model=RECOMMENDATION_MODEL,
    messages=[{"role": "user", "content": "Return ONLY this JSON and nothing else: {\"msg\": \"hi\"}"}],
    max_tokens=2000,
)

msg = r.choices[0].message
print("finish:", r.choices[0].finish_reason)
print("content:", repr(getattr(msg, "content", None)))
print("reasoning:", repr(getattr(msg, "reasoning", None))[:500] if getattr(msg, "reasoning", None) else None)
print("reasoning_content:", repr(getattr(msg, "reasoning_content", None))[:500] if getattr(msg, "reasoning_content", None) else None)