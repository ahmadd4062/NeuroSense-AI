from django.contrib import admin
from .models import QuestionnaireResult


@admin.register(QuestionnaireResult)
class QuestionnaireResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "final_prediction",
        "internal_model_score",
        "module_confidence",
        "confidence_level",
        "motor_score",
        "response_reliability",
        "created_at",
    )

    list_filter = (
        "final_prediction",
        "confidence_level",
        "reliability_level",
        "requires_fusion_review",
        "created_at",
    )

    search_fields = (
        "final_prediction",
        "confidence_level",
    )

    ordering = ("-created_at",)

    readonly_fields = (
        "created_at",
    )