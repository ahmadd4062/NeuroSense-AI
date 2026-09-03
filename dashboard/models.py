# dashboard/models.py

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class ScreeningHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='screenings')
    screening_id = models.CharField(max_length=64, unique=True)
    
    # Overall results
    risk_level = models.CharField(max_length=20)  # Low / Medium / High
    final_pd_probability = models.FloatField()
    fusion_confidence = models.FloatField()
    
    # Individual module results
    gait_prediction = models.CharField(max_length=20, null=True)
    speech_prediction = models.CharField(max_length=20, null=True)
    questionnaire_prediction = models.CharField(max_length=20, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d')}"