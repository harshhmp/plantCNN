from django.db import models
from django.contrib.auth.models import User
from .utils import getModelNames

# Create your models here.

class UserClassifications(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='media/')
    result = models.CharField(max_length=100)
    confidence = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    info = models.TextField(default="N/A")
    model = models.PositiveIntegerField(default=0)
    
    def __str__(self):
        return f"{self.user.userame} - {self.result}"
    
    def getModelName(self):
        return getModelNames()[self.model]