from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

urlpatterns = [
    path('', include('imageapp.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
