from django.urls import path
from . import views

app_name = 'stocknews'


urlpatterns = [
    path('news_home/', views.news_home, name="news_home"),
]