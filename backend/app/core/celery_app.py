import os
from unittest.mock import MagicMock
from celery import Celery

from app.core.config import settings

# Allow overriding broker/backend for testing
broker_url = os.environ.get("CELERY_BROKER_URL", settings.CELERY_BROKER_URL)
backend_url = os.environ.get("CELERY_RESULT_BACKEND", settings.CELERY_RESULT_BACKEND)

# For testing with memory broker, use a mock
if broker_url == "memory://" or backend_url == "memory://":
    # Create a mock Celery app for testing
    celery_app = MagicMock()
    celery_app.conf = {}
    
    def mock_task_decorator(*args, **kwargs):
        def decorator(func):
            mock_task = MagicMock()
            mock_task.__name__ = func.__name__
            mock_task.__module__ = func.__module__
            mock_task.__qualname__ = func.__qualname__
            mock_task.delay = MagicMock(return_value=None)
            mock_task.apply_async = MagicMock(return_value=None)
            return mock_task
        return decorator
    
    celery_app.task = mock_task_decorator
else:
    celery_app = Celery(
        "rag_tasks",
        broker=broker_url,
        backend=backend_url,
        include=["app.tasks.document_tasks"]
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )
