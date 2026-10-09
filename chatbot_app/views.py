import json
import logging
import traceback

from django.views.decorators.csrf import ensure_csrf_cookie
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .services.llm_service import get_llm_response
from .services.memory_extractor import extract_and_save_memory
from .services.safety_service import (
    check_emergency,
    get_emergency_response,
)
from .services.memory_service import (
    get_user_memories,
    get_or_create_chat_session,
    save_message,
    get_recent_messages,
)


logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def chatbot_page(request):
    return render(request, "chatbot_app/chatbot.html", {
        "hide_floating_chatbot": True,
    })


@require_POST
def chat_api(request):
    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()

        if not message:
            return JsonResponse(
                {
                    "success": False,
                    "error": "Please enter a message."
                },
                status=400
            )

        session = get_or_create_chat_session(request)

        conversation_history = get_recent_messages(
            session,
            limit=8
        )

        user_name = None
        user_memories = []

        if request.user.is_authenticated:
            user_name = request.user.first_name or request.user.username
            user_memories = get_user_memories(request.user)

        safety_result = check_emergency(message)

        if safety_result["is_emergency"]:
            emergency_reply = get_emergency_response()

            save_message(session, "user", message)
            save_message(session, "assistant", emergency_reply)

            if request.user.is_authenticated:
                extract_and_save_memory(
                    request.user,
                    message
                )

            return JsonResponse(
                {
                    "success": True,
                    "message": message,
                    "reply": emergency_reply,
                    "emergency": True,
                    "detected_flags": safety_result["detected_flags"],
                }
            )

        reply = get_llm_response(
            message=message,
            conversation_history=conversation_history,
            user_name=user_name,
            user_memories=user_memories
        )

        save_message(session, "user", message)
        save_message(session, "assistant", reply)

        if request.user.is_authenticated:
            extract_and_save_memory(
                request.user,
                message
            )

        return JsonResponse(
            {
                "success": True,
                "message": message,
                "reply": reply,
                "emergency": False,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {
                "success": False,
                "error": "Invalid request format."
            },
            status=400
        )

    except Exception as error:
        # ============================================================
        # TEMPORARY DEBUG BLOCK
        # Logs the full traceback to Django/Azure logs, prints to stdout,
        # and exposes the error details in the JSON response so it can be
        # read directly from the browser Network tab.
        #
        # REMOVE the `debug_error_type` and `debug_error_message` fields
        # after debugging is complete.
        # ============================================================
        error_type = type(error).__name__
        error_message = str(error)
        full_traceback = traceback.format_exc()

        logger.exception(f"Chatbot error: {error_message}")

        print("=== CHATBOT ERROR ===")
        print(f"Type: {error_type}")
        print(f"Message: {error_message}")
        print(f"Traceback:\n{full_traceback}")
        print("=== END CHATBOT ERROR ===")

        return JsonResponse(
            {
                "success": False,
                "error": "Something went wrong while processing your message.",
                "debug_error_type": error_type,
                "debug_error_message": error_message,
            },
            status=500
        )