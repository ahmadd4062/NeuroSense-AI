from django.contrib import admin
from .models import GaitResult


@admin.register(GaitResult)
class GaitResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "screening_id",
        "prediction",
        "pd_probability",
        "control_probability",
        "model_used",
        "pose_quality_score",
        "created_at",
    )

    list_filter = (
        "prediction",
        "model_used",
        "created_at",
    )

    search_fields = (
        "screening_id",
        "prediction",
        "user__username",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    ordering = ("-created_at",)

    fieldsets = (
        ("Prediction Results", {
            "fields": (
                "user",
                "screening_id",
                "prediction",
                "pd_probability",
                "control_probability",
                "model_used",
            )
        }),
        ("Pose Data", {
            "fields": (
                "pose_sequence",
            ),
            "classes": ("collapse",),
        }),
        ("Metadata", {
            "fields": (
                "video_duration",
                "video_fps",
                "pose_quality_score",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )