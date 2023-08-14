from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
import requests

# Create your views here.
def index(request):
    return render(request, "StockMarketApp/index.html", {})

def home(request):
    if request.method == 'POST':
        stock_name = request.POST.get('stock_search')
        time_frame = request.POST.get('time_frame')

        if not stock_name or not time_frame:
            print('Empty')
        else:
            print(time_frame)
            call_api(time_frame, stock_name)

        return redirect("/home")
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})

def suggest(request, symbol, time):
    call_api(time, symbol)
    return render(request, "StockMarketApp/candlestick_chart.html", {})

def call_api(time,stock):
    url = ''
    if time == '1':
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    elif time == '7':
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    elif time == '30':
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    else:
        print('error')

    if url:
        response = requests.get(url)
        data = response.json()

        if 'Error Message' in data:
            print('Invalid Stock Symbol')
        else:
            print(data)