from django.db import models

class GaitResult(models.Model):
    screening_id = models.CharField(max_length=64, unique=True)
    prediction = models.CharField(max_length=20)
    pd_probability = models.FloatField()
    control_probability = models.FloatField()
    model_used = models.CharField(max_length=50, default='LSTM')
    
    # NEW: Store pose sequence
    pose_sequence = models.JSONField(null=True, blank=True) 
    
    # Metadata
    video_duration = models.FloatField(null=True, blank=True)  
    video_fps = models.FloatField(null=True, blank=True)
    pose_quality_score = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'gait_results'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.screening_id} - {self.prediction}"