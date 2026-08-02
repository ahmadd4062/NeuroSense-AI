from django.contrib import admin

from .models import (
    RecordingPrediction,
    ScreeningPrediction,
)


class RecordingPredictionInline(admin.TabularInline):
    model = RecordingPrediction
    extra = 0
    can_delete = False
    readonly_fields = (
        "slot_number",
        "filename",
        "decision_score",
        "predicted_label",
        "probability",
        "prediction_successful",
        "feature_count",
        "error",
        "created_at",
    )


@admin.register(ScreeningPrediction)
class ScreeningPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "screening_id",
        "final_score",
        "final_label",
        "successful_recordings",
        "failed_recordings",
        "created_at",
    )

    list_filter = (
        "final_label",
        "created_at",
    )

    search_fields = (
        "screening_id",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
        "fusion_data",
        "metadata",
    )

    inlines = [
        RecordingPredictionInline,
    ]


@admin.register(RecordingPrediction)
class RecordingPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "screening",
        "slot_number",
        "decision_score",
        "predicted_label",
        "prediction_successful",
        "created_at",
    )

    list_filter = (
        "predicted_label",
        "prediction_successful",
    )

    search_fields = (
        "screening__screening_id",
        "filename",
    )