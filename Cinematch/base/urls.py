from django.urls import path
from .views import MainView

urlpatterns = [
    path('api/<str:userid>/', MainView, name='main'),
]