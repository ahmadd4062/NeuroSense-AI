import json

from django.conf import settings
from groq import Groq

from .memory_service import save_user_memory


client = Groq(api_key=settings.GROQ_API_KEY)


MEMORY_PROMPT = """
You are a memory extraction component for NeuroSense Assistant.

Your task is to decide whether the user's message contains useful long-term information
that may help personalize future conversations.

Only save important, reasonably stable information such as:
- recurring symptoms or concerns
- duration or pattern of an ongoing symptom
- completed NeuroSense screening
- important preferences related to guidance
- relevant ongoing context that may matter in future conversations

Do NOT save:
- greetings
- thanks
- general questions
- temporary statements
- random details
- assistant responses
- information that is not useful for future personalization

Return ONLY valid JSON in this format:

{
  "should_save": true,
  "category": "symptom",
  "memory_text": "User previously reported right-hand shaking that is more noticeable while resting."
}

If nothing should be saved:

{
  "should_save": false,
  "category": "",
  "memory_text": ""
}
"""


def extract_and_save_memory(user, message):
    if not user or not user.is_authenticated:
        return None

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": MEMORY_PROMPT
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        result = json.loads(
            response.choices[0].message.content
        )

        if not result.get("should_save"):
            return None

        category = result.get("category", "").strip()
        memory_text = result.get("memory_text", "").strip()

        if not category or not memory_text:
            return None

        return save_user_memory(
            user=user,
            category=category,
            memory_text=memory_text
        )

    except Exception as error:
        print(f"Memory extraction error: {error}")
        return None