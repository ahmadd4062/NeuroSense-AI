from django.db import models
from django.contrib.auth.models import User


class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    screening_id = models.CharField(max_length=64)
    content = models.TextField()
    risk_level = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'recommendations'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.screening_id} - {self.risk_level}"