from django.urls import path
from . import views

urlpatterns = [
    path('create_summary/', views.create_summary),
    path('update_emotion/', views.update_emotion),
    path('get_relevance/', views.speaker_relevance),
]
