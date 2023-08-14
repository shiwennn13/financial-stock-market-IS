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

        if not stock_name or time_frame:
            print('Empty')
        else:
            url = ''
            if time_frame == 1:
                url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ stock_name +'&apikey=C8W51TVO4E4FZ3HZ'
            elif time_frame == 7:
                url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+ stock_name +'&apikey=C8W51TVO4E4FZ3HZ'
            elif time_frame == 30:
                url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+ stock_name +'&apikey=C8W51TVO4E4FZ3HZ'
            else:
                print('error')

            if not url:
                response = requests.get(url)
                data = response.json()

                if data['Error Message']:
                    print('Invalid Stock Symbol')
                else:
                    print(data)

        return redirect("/home")
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})

def search(request):
    return render(request, "StockMarketApp/candlestick_chart.html", {})