from django.urls import path
from Movies import views as v

urlpatterns = [
    path('',v.test),
    path('api/home_movies',v.indexData),
]