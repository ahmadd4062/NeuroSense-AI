from django.db import models
from django.contrib.auth.models import User


class FusionResult(models.Model):
    """Store fused results from all three modules"""
    
    # Link to user
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='fusion_results')
    
    # Screening ID to link all modules
    screening_id = models.CharField(max_length=64, unique=True)
    
    # Individual module results
    gait_prediction = models.CharField(max_length=20, null=True, blank=True)
    gait_pd_probability = models.FloatField(null=True, blank=True)
    gait_control_probability = models.FloatField(null=True, blank=True)
    
    speech_prediction = models.CharField(max_length=20, null=True, blank=True)
    speech_pd_probability = models.FloatField(null=True, blank=True)
    speech_control_probability = models.FloatField(null=True, blank=True)
    
    questionnaire_prediction = models.CharField(max_length=20, null=True, blank=True)
    questionnaire_pd_probability = models.FloatField(null=True, blank=True)
    questionnaire_control_probability = models.FloatField(null=True, blank=True)
    
    # Fusion results
    final_prediction = models.CharField(max_length=20)  # 'PD' or 'Control'
    final_pd_probability = models.FloatField()
    final_control_probability = models.FloatField()
    fusion_confidence = models.FloatField()  # Max probability
    risk_level = models.CharField(max_length=20)  # 'Low', 'Medium', 'High'
    
    # Weights used
    gait_weight = models.FloatField(default=0.45)
    speech_weight = models.FloatField(default=0.35)
    questionnaire_weight = models.FloatField(default=0.20)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'fusion_results'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.screening_id} - {self.final_prediction} ({self.final_pd_probability:.3f})"