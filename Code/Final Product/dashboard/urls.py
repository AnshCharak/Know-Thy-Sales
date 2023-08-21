from django.contrib import admin
from django.urls import path
from dashboard.views import *

urlpatterns = [
    path('', analytics),
    path('results/', results),
]