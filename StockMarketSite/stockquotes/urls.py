from django.urls import path
from . import views


app_name = 'stockquotes'

urlpatterns = [
    path('qhome/', views.qhome, name="qhome"),
    path('add_stock.html', views.add_stock, name="add_stock"),
    path('delete/<stock_id>', views.delete, name="delete"),
]