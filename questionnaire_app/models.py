from django.db import models

class QuestionnaireResult(models.Model):
    final_prediction = models.CharField(
        max_length=50
    )

    module_confidence = models.FloatField()

    confidence_level = models.CharField(
        max_length=20
    )

    motor_score = models.FloatField()

    motor_pattern = models.CharField(
        max_length=20
    )

    positive_motor_symptoms = models.IntegerField()

    speech_cognition_score = models.FloatField()

    secondary_support_level = models.CharField(
        max_length=20
    )

    response_reliability = models.FloatField()

    reliability_level = models.CharField(
        max_length=20
    )

    uncertain_answers = models.IntegerField()

    age_in_model_range = models.BooleanField()

    internal_model_score = models.FloatField()

    internal_model_zone = models.CharField(
        max_length=20
    )

    requires_fusion_review = models.BooleanField(
        default=False
    )

    patient_interpretation = models.TextField()

    safety_flags = models.JSONField(
        default=list,
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return (
            f"{self.final_prediction} "
            f"({self.confidence_level})"
        )