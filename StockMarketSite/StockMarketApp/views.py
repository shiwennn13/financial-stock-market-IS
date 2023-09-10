import numpy
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect
import requests
import plotly.graph_objects as go
from datetime import datetime

from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt


# Create your views here.
def index(request):
    return render(request, "StockMarketApp/index.html", {})


def home(request):
    if request.method == 'POST':
        if 'search' in request.POST:
            stock_name = request.POST.get('stock_search')
            time_frame = request.POST.get('time_frame')

            if not stock_name or not time_frame:
                messages.error(request, 'Symbol or Time frame is empty')
            else:
                plot_div = call_api(request, time_frame, stock_name)

            return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': plot_div})

        else:
            symbol = ''
            time = ''
            if 'AAPL' in request.POST:
                symbol = "AAPL"
                time = '1'
            elif 'TSLA' in request.POST:
                symbol = "TSLA"
                time = '1'
            elif 'SBUX' in request.POST:
                symbol = "SBUX"
                time = '1'
            elif 'META' in request.POST:
                symbol = "META"
                time = '1'
            elif 'NKE' in request.POST:
                symbol = "NKE"
                time = "1"

            plot_div = call_api(request, time, symbol)
            return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': plot_div})
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})


def call_api(request, time, stock):
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
        messages.error(request, 'Symbol or Time frame is empty')

    if url:
        response = requests.get(url)
        data = response.json()

        if 'Error Message' in data:
            # print('Invalid Stock Symbol')
            messages.error(request, 'Invalid Stock Symbol')
        elif 'Note' in data:
            messages.error(request, 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 100 calls per day.')
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

                if time != '1':
                    ratio = float(data["5. adjusted close"]) / float(data["4. close"])
                    ratio_adjusted = float(data["1. open"]) / float(data["4. close"])

                    if (ratio * 100) < 80:
                        if ratio_adjusted * 100 > 150:
                            open_price = float(data["5. adjusted close"])
                            high_price = float(data["5. adjusted close"])
                            low_price = float(data["5. adjusted close"])
                        else:
                            open_price = ratio * float(data["1. open"])
                            high_price = ratio * float(data["2. high"])
                            low_price = ratio * float(data["3. low"])

                    else:
                        if (ratio_adjusted * 100) > 150:
                            open_price = float(data["5. adjusted close"])
                            high_price = float(data["5. adjusted close"])
                            low_price = float(data["5. adjusted close"])
                        else:
                            open_price = float(data["1. open"])
                            high_price = float(data["2. high"])
                            low_price = float(data["3. low"])

                    # high_price = float(data["2. high"])
                    # low_price = float(data["3. low"])
                    close_price = float(data["5. adjusted close"])

                    dates.append(date)
                    open.append(open_price)
                    high.append(high_price)
                    low.append(low_price)
                    close.append(close_price)
                else:
                    open_price = float(data["1. open"])
                    high_price = float(data["2. high"])
                    low_price = float(data["3. low"])
                    close_price = float(data["4. close"])

                    dates.append(date)
                    open.append(open_price)
                    high.append(high_price)
                    low.append(low_price)
                    close.append(close_price)

            lstm(close, time)

            return chart(open, high, low, close, dates)


def chart(open, high, low, close, date):
    # def chart():
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

def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []

    for i in range(len(dataset) - time_step - 1):
        a = dataset[i: (i+time_step), 0]
        # print(a)
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])

    return numpy.array(dataX), numpy.array(dataY)

def lstm(close, time):
    #scale and standardize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    #numpy.array = reshape from array into 2D array
    close = scaler.fit_transform(numpy.array(close).reshape(-1,1))
    # training size = 65%
    training_size = int(len(close)*0.65)

    # test size = 35%
    test_size = len(close)-training_size

    # split train data and test data from training_size
    train_data, test_data = close[0:training_size, :], close[training_size:len(close), :1]

    # use how many days data to predict, train data only have 65 data
    if time == "1":
        time_step = 2
    elif time == "7":
        time_step = 30
    else:
        time_step = 5

    # reshape into X = t , t+1 , t+2 and Y = t+3 meaning use t , t+1 , t+2 to predict t+3 according to time step.
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # print(X_train.shape)
    # print(X_train.shape[0])
    print(X_train.shape[1])
    # print(Y_train.shape)

    # print(X_test.shape)
    # print(Y_test.shape)

    #reshape input to become [samples, time steps, features] - required by LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #three layer of LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(LSTM(50))

    lstm_model.add(Dense(1))






