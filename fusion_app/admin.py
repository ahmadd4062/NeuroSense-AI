from django.contrib import admin
from .models import FusionResult


@admin.register(FusionResult)
class FusionResultAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'screening_id',
        'final_prediction',
        'final_pd_probability',
        'fusion_confidence',
        'risk_level',
        'created_at'
    )
    list_filter = ('final_prediction', 'risk_level', 'created_at')
    search_fields = ('screening_id', 'user__username')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-created_at',)