import numpy
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import CustomUserCreationForm
# from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.http import HttpResponseRedirect
import requests
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import matplotlib.pyplot as plt
import tensorflow as tf
import math


# Create your views here.

def user_login ( request ) :
    page = 'login'
    if request.method == 'POST' :
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None :
            login(request, user)
            return redirect('index')  # Redirect to the home page or dashboard
        else :
            # Invalid login
            return render(request, 'StockMarketApp/login.html', {'error' : 'Invalid username or password','page': page})
    else :
        # Handle GET request
        return render(request, 'StockMarketApp/login.html', {'page': page})


def user_logout ( request ) :
    logout(request)
    return redirect('login')


def user_register ( request ) :
    page = 'register'
    form = CustomUserCreationForm()
    if request.method == 'POST' :
        form = CustomUserCreationForm(request.POST)
        if form.is_valid() :
            user = form.save(commit=False)
            user.save()
            user = authenticate(request, username=user.username, password=request.POST['password1'])

            if user is not None :
                login(request, user)
                messages.success(request, 'Your registration was successful. Welcome!')
                return redirect('index')
        else :
            # Handle form errors here, if needed
            messages.error(request, 'There was a problem with your registration. Please correct the errors below.')

    context = {'form' : form, 'page' : page}
    return render(request, 'StockMarketApp/login.html', context)


def index ( request ) :
    return render(request, "StockMarketApp/index.html", {})


@login_required(login_url='login')
def prediction ( request ) :
    if request.method == 'POST' :
        if 'search' in request.POST :
            stock_name = request.POST.get('stock_search')
            time_frame = request.POST.get('time_frame')

            if not stock_name or not time_frame :
                messages.error(request, 'Symbol or Time frame is empty')
            else :
                plot_div, plot_div2 = call_api(request, time_frame, stock_name)

            return render(request, "StockMarketApp/candlestick_chart.html",
                          {'plot_div' : plot_div, 'plot_div2' : plot_div2})

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

            plot_div, plot_div2 = call_api(request, time, symbol)

            return render(request, "StockMarketApp/candlestick_chart.html", {'plot_div': plot_div, 'plot_div2':plot_div2 })
    else:
        return render(request, "StockMarketApp/candlestick_chart.html", {})


def call_api(request, time, stock):
    url = ''
    day = ''
    if time == '1':
        day = 'Time Series (Daily)'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + stock + '&apikey=0BA8N0PMQV2APZGB'
    elif time == '7':
        day = 'Weekly Adjusted Time Series'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=' + stock + '&apikey=0BA8N0PMQV2APZGB'
    elif time == '30':
        day = 'Monthly Adjusted Time Series'
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stock + '&apikey=0BA8N0PMQV2APZGB'
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

            train, test, prediction = lstm(close, time)

            return chart(open, high, low, close, dates, train, test, prediction, time)


def chart(open, high, low, close, date, train, test, prediction, time):
    # def chart():
    open_data = open
    high_data = high
    low_data = low
    close_data = close
    dates = date

    # print(test)
    # print(prediction)

    train = numpy.array(train)
    train = train.ravel()
    train = ', '.join(map(str, train))
    train = numpy.fromstring(train, sep=',')
    train = train[::-1]

    # print(train)
    print(time)

    test = numpy.array(test)
    test = test.ravel()
    test = ', '.join(map(str, test))
    test = numpy.fromstring(test, sep=',')
    test = test[::-1]

    pred = numpy.array(prediction)
    pred = pred.ravel()
    pred = ', '.join(map(str, pred))
    pred = numpy.fromstring(pred, sep=',')

    test_date = dates[0]

    datess = []

    if time == '1':
        for _ in range(30):
            datess.append(test_date)
            test_date += timedelta(days=1)
    elif time == '7':
        for _ in range(30):
            datess.append(test_date)
            test_date += timedelta(days=7)
    else:
        for _ in range(30):
            datess.append(test_date)
            test_date += timedelta(days=30)

    print(test)
    print(pred)

    # print(dates)

    # Calculate the highest and lowest values in high_data and low_data
    y_max = max(max(high_data), max(low_data))
    y_min = min(min(high_data), min(low_data))

    candle_stick = go.Candlestick(x=dates, open=open_data, high=high_data, low=low_data, close=close_data)

    line = go.Scatter(x=dates, y=close_data, name='Line Chart')

    line_train = go.Scatter(x=dates, y=train, name='Line Chart Train')

    line_test = go.Scatter(x=dates, y=test, name='Line Chart Test')

    line_prediction = go.Scatter(x=datess, y=pred, name='Line Chart Prediction')

    line_fig = go.Figure(data=[line, line_train, line_test, line_prediction])

    fig = go.Figure(data=[candle_stick])

    # Set the y-axis range to fit the data
    fig.update_yaxes(range=[y_min - 1, y_max + 1])  # Add some padding for clarity

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, width=1080, yaxis_autorange=True)

    line_fig.update_layout(height=600, width=1080, yaxis_autorange=True)

    config = {'displayModeBar': False}

    div = fig.to_html(full_html=False, config=config)

    div2 = line_fig.to_html(full_html=False, config=config)

    return div, div2

def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []

    for i in range(len(dataset) - time_step - 1):
        a = dataset[i: (i+time_step), 0]
        # print(a)
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])

    return numpy.array(dataX), numpy.array(dataY)

def lstm(close, time):
    close = close[::-1]
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
        epoch = 100
        batch_size = 128
    elif time == "7":
        time_step = 30
        epoch = 100
        batch_size = 64
    else:
        time_step = 15
        epoch = 100
        batch_size = 64

    # reshape into X = t , t+1 , t+2 and Y = t+3 meaning use t , t+1 , t+2 to predict t+3 according to time step.
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    #reshape input to become [samples, time steps, features] - required by LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #three layer of LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(LSTM(50))

    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')

    #view model summary
    # print(lstm_model.summary())

    lstm_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epoch, batch_size=batch_size, verbose=1)

    #prediction occur and check performance metrics
    train_predict = lstm_model.predict(X_train)
    test_predict = lstm_model.predict(X_test)

    #from scaler - tranform back to normal
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    #train data RMSE
    train_data_rmse = math.sqrt(mean_squared_error(Y_train,train_predict))

    #test data RMSE
    test_data_rmse = math.sqrt(mean_squared_error(Y_test, test_predict))

    # print(train_data_rmse)
    # print(test_data_rmse)
    trainPredictPlot = numpy.empty_like(close)
    trainPredictPlot[:,:] = numpy.nan
    trainPredictPlot[time_step: len(train_predict)+time_step, :] = train_predict

    testPredictPlot = numpy.empty_like(close)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(time_step*2)+1:len(close)-1, :] = test_predict

    x_input_len = len(test_data) * 0.2
    x_input_lens = len(test_data) - round(x_input_len)

    x_input = test_data [x_input_lens:].reshape(1,-1)
    # print(x_input.shape)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # print(temp_input)

    lst_output = []
    n_steps = round(x_input_len)
    # print(n_steps)
    i = 0

    while (i<30): # 30 days
        if(len(temp_input)>n_steps):
            #print temp input
            x_input=numpy.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1,n_steps, 1))
            #print x input
            yhat = lstm_model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print temp input
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = lstm_model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    # print(lst_output)
    prediction = scaler.inverse_transform(lst_output)

    return trainPredictPlot, testPredictPlot, prediction














