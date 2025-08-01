from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserClassifications(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='media/')
    result = models.CharField(max_length=100)
    confidence = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.userame} - {self.result}"