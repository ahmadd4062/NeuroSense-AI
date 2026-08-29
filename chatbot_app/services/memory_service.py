from chatbot_app.models import ChatSession, ChatMessage


def get_or_create_chat_session(request):
    if request.user.is_authenticated:
        session = (
            ChatSession.objects
            .filter(user=request.user)
            .order_by("-updated_at")
            .first()
        )

        if session:
            return session

        return ChatSession.objects.create(user=request.user)

    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    session, _ = ChatSession.objects.get_or_create(
        user=None,
        session_key=session_key
    )

    return session


def save_message(session, role, content):
    return ChatMessage.objects.create(
        session=session,
        role=role,
        content=content
    )


def get_recent_messages(session, limit=8):
    messages = list(
        session.messages
        .order_by("-created_at")[:limit]
    )

    messages.reverse()

    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]

from chatbot_app.models import UserMemory


def get_user_memories(user, limit=10):
    if not user or not user.is_authenticated:
        return []

    memories = UserMemory.objects.filter(
        user=user
    ).order_by("-updated_at")[:limit]

    return [
        {
            "category": memory.category,
            "memory_text": memory.memory_text
        }
        for memory in memories
    ]


def save_user_memory(user, category, memory_text):
    if not user or not user.is_authenticated:
        return None

    existing = UserMemory.objects.filter(
        user=user,
        category=category,
        memory_text__iexact=memory_text
    ).first()

    if existing:
        return existing

    return UserMemory.objects.create(
        user=user,
        category=category,
        memory_text=memory_text
    )