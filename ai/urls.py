from django.urls import path
from . import views
from django.urls import re_path
from . import consumers

urlpatterns = [
    path('create_summary/', views.create_summary),
    path('update_emotion/', views.update_emotion),
    path('get_relevance/', views.speaker_relevance),
  path('analyze_transcription/', views.analyze_transcription, name='analyze_transcription'),
   re_path(r'ws/progress/(?P<task_id>\w+)/$', consumers.ProgressConsumer.as_asgi()),
]
