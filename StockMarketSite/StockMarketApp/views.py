from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "StockMarketApp/candlestick_chart.html", {})
