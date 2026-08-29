import re

from django.conf import settings
from groq import Groq

from .rag_service import retrieve_relevant_chunks


client = Groq(api_key=settings.GROQ_API_KEY)


SYSTEM_PROMPT = """
You are NeuroSense Assistant, a conversational health guidance assistant.

Follow these rules carefully:

- Answer the user's current concern directly and naturally.
- Do not mention Parkinson's unless the user asks about Parkinson's or the context clearly relates to it.
- Do not give unrelated precautions or generic advice.
- Do not diagnose any disease.
- Give concise, practical, scenario-relevant guidance.
- When follow-up questions are useful, ask only 2 to 4 short, relevant questions in the same response.
- Do not repeatedly question the user without giving useful guidance.
- If the user reports a serious or urgent symptom, advise urgent professional medical care.
- For greetings, thanks, acknowledgements, and very short conversational messages, respond briefly and do not unnecessarily repeat previous medical advice.
- Keep the tone friendly, calm, and conversational.
- Avoid long lists unless absolutely necessary.
- Prefer short paragraphs over lengthy explanations.

- When relevant NeuroSense knowledge-base context is provided, prioritize it for factual Parkinson's-related information.
- Never treat retrieved information as proof that the user has Parkinson's or any other condition.

- Never prescribe, recommend, select, start, stop, replace, or change the dose or timing of any medication.
- Do not tell the user which medication they should take.
- You may provide general educational information about medications when explicitly relevant, but medication decisions must be referred to an appropriate healthcare professional.

- Always treat the user's latest message as the primary request.
- Previous conversation may be provided only when the latest message appears to depend on it.
- Never continue discussing an older symptom when the user has clearly asked a new question.

- Do not recommend specific medical interventions, supplements, dietary changes, devices, or therapeutic exercises unless they are clearly supported by the provided knowledge-base context.
- When evidence or user context is insufficient, provide general guidance and recommend appropriate professional evaluation instead of inventing a management plan.
- Do not invent specific durations, frequencies, quantities, doses, targets, or step-by-step medical/self-management instructions unless they are explicitly supported by the provided NeuroSense knowledge-base context.
- When refusing medication selection, dosage, starting, stopping, or changing treatment, keep the response brief. Do not collect detailed medication history unless the user specifically asks for help discussing treatment with a healthcare professional.
- The chatbot supports English, Urdu, and Roman Urdu.
- Detect the language of the user's latest message and respond in the same language.
- If the user writes in English, respond in English.
- If the user writes in Urdu script, respond in Urdu script.
- If the user writes in Roman Urdu, respond naturally in Roman Urdu.
- If the user mixes English and Urdu, respond in a similar natural bilingual style.
- Keep medical terminology understandable. When an Urdu medical term may be unclear, include the common English term in parentheses when helpful.
- Do not translate the user's message unless they explicitly ask for a translation.
"""


def should_use_conversation_history(message):
    text = message.lower().strip()

    follow_up_patterns = [
        r"\bit\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\bthose\b",
        r"\bthese\b",
        r"\bstill\b",
        r"\bagain\b",
        r"\bsame\b",
        r"\bsince then\b",
        r"\bafter that\b",
        r"\bstarted yesterday\b",
        r"\bstarted today\b",
        r"\bgot worse\b",
        r"\bgetting worse\b",
        r"\bgetting better\b",
        r"\bfeels better\b",
        r"\bfeels worse\b",
        r"\bhappens mostly\b",
        r"\bhappens when\b",
        r"\bonly happens\b",
        r"\bfor a few seconds\b",
        r"\bfor a few minutes\b",
        r"\bone hand\b",
        r"\bboth hands\b",
    ]

    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in follow_up_patterns
    ):
        return True

    words = text.split()

    if len(words) <= 8 and not text.endswith("?"):
        return True

    return False

def is_parkinsons_question(message):
    text = message.lower()

    terms = [
        "parkinson",
        "parkinson's",
        "parkinsons",
    ]

    return any(term in text for term in terms)
def has_grounded_rag_support(message, rag_results):
    if not rag_results:
        return False

    text = message.lower()

    stop_words = {
        "can", "could", "does", "do", "did",
        "is", "are", "was", "were",
        "a", "an", "the", "and", "or",
        "of", "to", "in", "on", "for",
        "with", "someone", "person",
        "people", "my", "me", "i",
        "have", "has", "had",
        "cause", "affect",
        "parkinson", "parkinsons"
    }

    query_words = {
        word
        for word in re.findall(r"\b[a-z]+\b", text)
        if word not in stop_words and len(word) > 2
    }

    if not query_words:
        return bool(rag_results)

    for result in rag_results:
        document_text = (
            f"{result.get('title', '')} "
            f"{result.get('topic', '')} "
            f"{result.get('content', '')}"
        ).lower()

        matched_words = {
            word
            for word in query_words
            if word in document_text
        }

        if matched_words:
            return True

    return False
def get_llm_response(
    message,
    conversation_history=None,
    user_name=None,
    user_memories=None
):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if user_name:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"The user's name is {user_name}. "
                    "Use their name naturally when appropriate, "
                    "but do not repeat it unnecessarily."
                )
            }
        )

    if user_memories:
        memory_text = "\n".join(
            f"- {memory['memory_text']}"
            for memory in user_memories
        )

        messages.append(
            {
                "role": "system",
                "content": (
                    "The following are selected long-term memories about "
                    "the user from previous conversations:\n"
                    f"{memory_text}\n\n"
                    "Use a memory only if it is directly relevant to the "
                    "user's latest message. Do not force old health concerns "
                    "into an unrelated conversation."
                )
            }
        )

    rag_results = retrieve_relevant_chunks(
        message,
        top_k=3
    )
    has_rag_context = bool(rag_results)
    has_grounded_context = has_grounded_rag_support(
        message,
        rag_results
    )
    if rag_results:
        rag_context = "\n\n".join(
            f"Organization: {result['organization']}\n"
            f"Title: {result['title']}\n"
            f"Topic: {result['topic']}\n"
            f"Content: {result['content']}"
            for result in rag_results
        )
    if is_parkinsons_question(message) and not has_grounded_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "The user is asking a Parkinson's-specific factual question, "
                    "but the retrieved NeuroSense knowledge-base context does not "
                    "clearly support the specific topic being asked about. "
                    "Do not invent or confidently state medical facts from general "
                    "model knowledge. Briefly explain that the current NeuroSense "
                    "knowledge base does not contain enough verified information "
                    "to answer this reliably, and suggest consulting an appropriate "
                    "healthcare professional or trusted medical source."
                )
            }
        )
    if is_parkinsons_question(message):
        if not has_rag_context:
         messages.append(
                {
                    "role": "system",
                    "content": (
                        "The user is asking a Parkinson's-specific factual "
                        "question, but the NeuroSense knowledge base did not "
                        "retrieve supporting information. Do not invent or "
                        "confidently state medical facts from general model "
                        "knowledge. Explain briefly that the current NeuroSense "
                        "knowledge base does not contain enough verified "
                        "information to answer this question reliably, and "
                        "suggest consulting an appropriate healthcare "
                        "professional or trusted medical source."
                    )
                }
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following information comes from the NeuroSense "
                    "curated knowledge base. Use only the information that "
                    "is relevant to the user's latest question. Do not use "
                    "retrieved information to claim or confirm a diagnosis.\n\n"
                    f"{rag_context}"
                )
            }
        )

    if (
        conversation_history
        and should_use_conversation_history(message)
    ):
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following recent conversation is provided because "
                    "the latest message appears to refer back to earlier context."
                )
            }
        )

        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.4,
        max_tokens=350,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        return (
            "I couldn't generate a reliable response just now. "
            "Please try asking again."
        )

    return content.strip()