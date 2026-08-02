from __future__ import annotations

from django.db import models


class ScreeningPrediction(models.Model):
    class ResultLabel(models.TextChoices):
        HEALTHY = "Healthy", "Healthy"
        BORDERLINE = "Borderline", "Borderline"
        PARKINSON = "Parkinson", "Parkinson"

    screening_id = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
    )

    final_score = models.FloatField()
    final_label = models.CharField(
        max_length=20,
        choices=ResultLabel.choices,
    )

    healthy_threshold = models.FloatField(default=0.23)
    parkinson_threshold = models.FloatField(default=0.27)

    successful_recordings = models.PositiveSmallIntegerField(default=0)
    failed_recordings = models.PositiveSmallIntegerField(default=0)

    # Optional full fusion output for debugging.
    fusion_data = models.JSONField(
        default=dict,
        blank=True,
    )

    # Optional request/session metadata.
    metadata = models.JSONField(
        default=dict,
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return (
            f"{self.screening_id} — "
            f"{self.final_label} ({self.final_score:.4f})"
        )


class RecordingPrediction(models.Model):
    screening = models.ForeignKey(
        ScreeningPrediction,
        on_delete=models.CASCADE,
        related_name="recording_predictions",
    )

    slot_number = models.PositiveSmallIntegerField()

    filename = models.CharField(
        max_length=255,
        blank=True,
    )

    audio_path = models.TextField(
        blank=True,
    )

    decision_score = models.FloatField(
        null=True,
        blank=True,
    )

    predicted_label = models.CharField(
        max_length=20,
        blank=True,
    )

    model_prediction = models.CharField(
        max_length=100,
        blank=True,
    )

    probability = models.FloatField(
        null=True,
        blank=True,
    )

    extraction_successful = models.BooleanField(default=False)
    prediction_successful = models.BooleanField(default=False)

    feature_count = models.PositiveIntegerField(default=0)

    error = models.TextField(
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["slot_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["screening", "slot_number"],
                name="unique_prediction_per_screening_slot",
            )
        ]

    def __str__(self) -> str:
        return (
            f"{self.screening.screening_id} "
            f"slot {self.slot_number}"
        )