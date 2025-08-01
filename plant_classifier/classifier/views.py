from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

from .forms import UploadImageForm
from .utils import getModel1ClassNames, getModel2ClassNames, runEfficientNet, runCustomModel
from .models import UserClassifications

from PIL import Image
import numpy as np
import tensorflow as tf

import os

# Create your views here.

# Load models to use in classifcation (move to utils.py later)
model1 = tf.keras.models.load_model("../model1-2.keras")
model2 = tf.keras.models.load_model("../model1-3.keras")

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
            
            # Prepare image for classification
            image = Image.open(uploaded_file).resize((128, 128))
            
            img_array = tf.keras.utils.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            if model_to_run == 'model1':
                # Run Model 1, and translate result into words
                prediction = model1.predict(img_array)
                confidence = float(prediction.max()) * 100
                
                class_index = prediction.argmax()
                
                CLASS_NAMES = getModel1ClassNames()
                result = CLASS_NAMES[class_index]
                print("ran model1")
                
            elif model_to_run == 'model2':
                # Run Model 2, and translate result into words
                prediction = model2.predict(img_array)
                confidence = float(prediction.max()) * 100
                
                class_index = prediction.argmax()
                
                CLASS_NAMES = getModel2ClassNames()
                result = CLASS_NAMES[class_index]
                print("ran model2")
            
            elif model_to_run == 'model3':
                # Run Model 3, and translate result into words
                image2 = Image.open(uploaded_file).resize((224, 224))
            
                img_array2 = tf.keras.utils.img_to_array(image2)
                img_array2 = np.expand_dims(img_array2, axis=0)
            
                decoded = runEfficientNet(img_array2)
                result = decoded[0][1]
                confidence = float(decoded[0][2]) * 100
            
            elif model_to_run == 'model4':
                # Run Model 4, and translate result into words
                image2 = Image.open(uploaded_file).resize((224, 224))
            
                img_array2 = tf.keras.utils.img_to_array(image2)
                img_array2 = np.expand_dims(img_array2, axis=0)
                
                confidence, result = runCustomModel(img_array2)
                
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