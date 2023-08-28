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
            return redirect("/home")
        else:
            plot_div = call_api(time_frame, stock_name)
            return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': plot_div})
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})

def suggest(request, symbol, time):
    plot_div = call_api(time, symbol)
    return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': plot_div})

def call_api(time,stock):
    url = ''
    day = ''
    if time == '1':
        day = 'Time Series (Daily)'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    elif time == '7':
        day = 'Weekly Adjusted Time Series'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    elif time == '30':
        day = 'Monthly Adjusted Time Series'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stock + '&apikey=C8W51TVO4E4FZ3HZ'
    else:
        print('error')

    if url:
        response = requests.get(url)
        data = response.json()

        if 'Error Message' in data:
            print('Invalid Stock Symbol')
        else:
            dates = []
            open = []
            high = []
            low = []
            close = []

            for date_str, data in data[day].items():
                date_parts = date_str.split("-")

                year = int(date_parts[0])
                month = int(date_parts[1])
                day = int(date_parts[2])

                date = datetime(year, month, day)

                ratio = float(data["5. adjusted close"]) / float(data["4. close"])
                ratio_adjusted = float(data["1. open"]) / float(data["4. close"])

                if (ratio * 100) < 80:
                    open_price = ratio * float(data["1. open"])
                    high_price = ratio * float(data["2. high"])
                    low_price = ratio * float(data["3. low"])

                else:
                    if (ratio_adjusted * 100) > 150:
                        open_price = float(data["5. adjusted close"])
                        high_price = float(data["5. adjusted close"])
                        low_price = float(data["3. low"])
                    else:
                        open_price = float(data["1. open"])
                        high_price = float(data["2. high"])
                        low_price = float(data["3. low"])

                #high_price = float(data["2. high"])
                #low_price = float(data["3. low"])
                close_price = float(data["5. adjusted close"])

                dates.append(date)
                open.append(open_price)
                high.append(high_price)
                low.append(low_price)
                close.append(close_price)

            return chart(open, high, low, close, dates)


def chart(open, high, low, close, date):
#def chart():
    open_data = open
    high_data = high
    low_data = low
    close_data = close
    dates = date

    # Calculate the highest and lowest values in high_data and low_data
    y_max = max(max(high_data), max(low_data))
    y_min = min(min(high_data), min(low_data))

    fig = go.Figure(data=[go.Candlestick(x=dates,
                                         open=open_data, high=high_data,
                                         low=low_data, close=close_data)])

    # Set the y-axis range to fit the data
    fig.update_yaxes(range=[y_min - 1, y_max + 1])  # Add some padding for clarity

    fig.update_layout(xaxis_rangeslider_visible=False, height=700, yaxis_autorange=True)

    config = {'displayModeBar': False}

    div = fig.to_html(full_html=False, config=config)

    return div
