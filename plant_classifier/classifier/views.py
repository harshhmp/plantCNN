from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

from .forms import UploadImageForm
from .utils import runEfficientNet, runCustomModel, runModel1, runModel2
from .models import UserClassifications

from PIL import Image
import numpy as np
import tensorflow as tf

import os

# Create your views here.

@login_required
def classify_image(request):
    # Initialize variables to send for rendering
    result = None
    confidence = None
    image_url = None
    form = UploadImageForm()
    
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['image']
            
            model_to_run = request.POST.get('mode', 'default')
            
            if model_to_run == 'model1':
                # Run Model 1, and translate result into words
                confidence, result = runModel1(uploaded_file)
                print("ran model1")
                
            elif model_to_run == 'model2':
                # Run Model 2, and translate result into words
                confidence, result = runModel2(uploaded_file)
                print("ran model2")
            
            elif model_to_run == 'model3':
                # Run Model 3, and translate result into words
                decoded = runEfficientNet(uploaded_file)
                result = decoded[0][1]
                confidence = float(decoded[0][2]) * 100
                print("ran efficient Model")
            
            elif model_to_run == 'model4':
                # Run Model 4, and translate result into words
                confidence, result = runCustomModel(uploaded_file)
                print("ran model4")

            # Create user history object to store image and result
            image = form.cleaned_data['image']
            record = UserClassifications.objects.create(
                user=request.user,
                image=image,
                result=result,
                confidence = confidence
            )
            
            image_url = record.image.url
        
    return render(request, 'classify.html', {
        'form': form,
        'result': result,
        'confidence': confidence,
        'image_url': image_url,
    })

def about_page(request):
    return render(request, 'about.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
            
    return render(request, 'signup.html', {'form': form})

@login_required
def history_view(request):
    # Find user's history sorted by date created
    records = UserClassifications.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'history.html', {'records': records})

@login_required
def delete_record(request, record_id):
    # Find object requested to delete and the image's path
    record = get_object_or_404(UserClassifications, id=record_id, user=request.user)
    full_path = os.path.join(settings.BASE_DIR , record.image.path)
    
    if request.method == 'POST':
        # Delete history record and return to history page
        default_storage.delete(full_path)
        record.delete()
        return redirect('history')