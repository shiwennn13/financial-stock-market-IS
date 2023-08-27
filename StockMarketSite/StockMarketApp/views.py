from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
import requests
import plotly.graph_objects as go
from datetime import datetime

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
            chart()

        return redirect("/home")
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})

def suggest(request, symbol, time):
    call_api(time, symbol)
    fig = chart()
    config = {'displayModeBar': False}
    div = fig.to_html(full_html=False, config=config)
    return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': div})

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

def chart():
    open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    dates = [datetime(year=2013, month=10, day=10),
             datetime(year=2013, month=11, day=10),
             datetime(year=2013, month=12, day=10),
             datetime(year=2014, month=1, day=10),
             datetime(year=2014, month=2, day=10)]

    fig = go.Figure(data=[go.Candlestick(x=dates,
                                         open=open_data, high=high_data,
                                         low=low_data, close=close_data)])

    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig
