from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect


# Create your views here.
def index(request):
    return render(request, "StockMarketApp/index.html", {})

def home(request):
    if request.method == 'POST':
    #    stock_name = request.POST.get('stock_search')
        return redirect("/index")
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})

def search(request):
    return render(request, "StockMarketApp/candlestick_chart.html", {})