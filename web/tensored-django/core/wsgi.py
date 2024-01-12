"""
WSGI config for core project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

# Import necessary modules
import os

# Import the WSGI application from Django
from django.core.wsgi import get_wsgi_application

# Set the default Django settings module to 'core.settings'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Create the WSGI application using Django's get_wsgi_application function
application = get_wsgi_application()
