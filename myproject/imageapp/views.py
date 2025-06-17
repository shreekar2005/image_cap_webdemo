from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from django.conf import settings
import os
from .test import process_image
from django.shortcuts import redirect

def index(request):
    return redirect('upload_image')

def upload_image(request):
    caption = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            image_path = os.path.join(settings.MEDIA_ROOT, instance.image.name)
            caption = process_image(image_path)
            # Delete the image file after processing
            if os.path.exists(image_path):
                os.remove(image_path)
            # Optionally, delete the database record as well
            instance.delete()
    else:
        form = ImageUploadForm()
    return render(request, 'imageapp/upload.html', {'form': form, 'caption': caption})
