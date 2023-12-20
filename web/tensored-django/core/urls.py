"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.urls import path
from .views import index, bcc, melanoma, nevus, references, privacy, specifications, about_me


urlpatterns = [
    path('', index, name='index'),
    path('admin/', admin.site.urls),
    path('bcc/', bcc, name='BCC'),
    path('melanoma/', melanoma, name='Melanoma'),
    path('nevus/', nevus, name='Nevus'),
    path('references/', references, name='References'),
    path('privacy/', privacy, name='Privacy'),
    path('specifications/', specifications, name='Specifications'),
    path('about_me/', about_me, name='AboutMe'),
]

if settings.DEBUG:
    # Serving media files during development
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # Serving static files during development
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
