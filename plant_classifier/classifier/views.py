from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.utils.safestring import mark_safe

from .forms import UploadImageForm
from .utils import runEfficientNet, runCustomModel, runModel1, runModel2, runCustomModel1, cleanText, getModelNames
from .models import UserClassifications

from PIL import Image
import numpy as np
import tensorflow as tf
from openai import OpenAI

import os
import json

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
            model_num = 0
            
            if model_to_run == 'model1':
                # Run Model 1, and translate result into words
                confidence, result = runModel1(uploaded_file)
                print("ran model1")
                model_num = 1
                
            elif model_to_run == 'model2':
                # Run Model 2, and translate result into words
                confidence, result = runModel2(uploaded_file)
                print("ran model2")
                model_num = 2
            
            elif model_to_run == 'model3':
                # Run Model 3, and translate result into words
                decoded = runEfficientNet(uploaded_file)
                result = decoded[0][1]
                confidence = float(decoded[0][2]) * 100
                print("ran efficient Model")
                model_num = 3
            
            elif model_to_run == 'model4':
                # Run Model 4, and translate result into words
                confidence, result = runCustomModel(uploaded_file)
                print("ran model4")
                model_num = 4
            
            elif model_to_run == "model5":
                confidence, result = runCustomModel1(uploaded_file)
                print("ran model5")
                model_num = 5

            # Create user history object to store image and result
            image = form.cleaned_data['image']
            record = UserClassifications.objects.create(
                user = request.user,
                image = image,
                result = result,
                confidence = confidence,
                model = model_num
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
    
def info_view(request, record_id):
    record = get_object_or_404(UserClassifications, id=record_id, user=request.user)
    
    if record.info == "N/A":
        # Generate Info here and change the record if information doesn't exist
        key = settings.OPEN_AI_KEY
        
        client = OpenAI(api_key=key)
        
        prompt = f"Give me some information on {record.result} in under 100 words. Format this using HTML with bootstrap and give me only the container div with no added fluff, start with <div...>"
        
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            store=True,
        )
        
        clean_response = cleanText(response.output_text)
        
        print("Generated: " + clean_response)
        
        record.info = mark_safe(clean_response)
        record.save()
        
    print("Retrieved: " + record.info)
    
    if request.method == 'POST':
        about = "No information for this model or classification, try using Defualt Model or Custom Model ++"
        
        # If applicable model is used to classify image, get information from json file
        if record.model == 5 or record.model == 1:
            with open('../plantData/plant diseases cure/cure.json') as f:
                data = json.load(f)
            
            key = record.result.lower()
            if key in data: 
                about = data[key]
                
        return render(request, 'info.html', {'record': record, 'response_text': record.info, 'about': about})