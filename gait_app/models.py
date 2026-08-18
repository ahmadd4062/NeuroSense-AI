from django.db import models
from django.contrib.auth.models import User

class GaitResult(models.Model):
    # Link to user (to match Fatima and Bushra's database)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='gait_results')
    
    screening_id = models.CharField(max_length=64, unique=True)
    prediction = models.CharField(max_length=20)  # 'PD' or 'Control'
    pd_probability = models.FloatField()
    control_probability = models.FloatField()
    model_used = models.CharField(max_length=50, default='LSTM')
    
    # Store pose sequence
    pose_sequence = models.JSONField(null=True, blank=True)  # (50, 34) array
    
    # Metadata
    video_duration = models.FloatField(null=True, blank=True)  # seconds
    video_fps = models.FloatField(null=True, blank=True)
    pose_quality_score = models.FloatField(null=True, blank=True)  # 0-1
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'gait_results'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.screening_id} - {self.prediction} ({self.pd_probability:.3f})"