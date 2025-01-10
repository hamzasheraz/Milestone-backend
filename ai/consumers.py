# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def send_progress(self, progress):
        await self.send(text_data=json.dumps({
            'progress': progress
        }))
