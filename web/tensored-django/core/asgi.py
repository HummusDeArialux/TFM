"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""

# Import necessary modules
import os

# Import the ASGI application from Django
from django.core.asgi import get_asgi_application

# Set the default Django settings module to 'core.settings'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Create the ASGI application using Django's get_asgi_application function
application = get_asgi_application()
