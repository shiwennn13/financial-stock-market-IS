from django.shortcuts import render, redirect
from .models import Stock
from .forms import StockForm
from django.contrib import messages


def qhome ( request ) :
    import requests
    import json

    processed_api = {}  # Initialize as an empty dictionary

    if request.method == 'POST' :
        ticker = request.POST['ticker']  # 'ticker' is the one in the qhome.html <input> name
        # apikey: AD3P4POXBTNNCS7D
        api_request = requests.get(
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + ticker + "&apikey"
                                                                                         "=AD3P4POXBTNNCS7D")

        try :
            api = json.loads(api_request.content)
            api_data = api.get('Global Quote', {})
            if not api_data :
                processed_api = "Error..."
            else :
                processed_api = {key.replace(' ', '_').replace('.', '') : value for key, value in api_data.items()}
        except Exception as e :
            processed_api = "Error..."
            # print("Exception occurred:", e)
            # api_error = True

        return render(request, 'stockquotes/qhome.html', {'api' : processed_api})
    else :
        return render(request, 'stockquotes/qhome.html', {'ticker' : "Enter a ticker symbol above"})

    # return render(request, 'stockquotes/qhome.html', {'api': processed_api})



def add_stock ( request ) :
    import requests
    import json

    if request.method == 'POST' :
        form = StockForm (request.POST or None)

        if form.is_valid():
            form.save()
            messages.success(request,("Stock Has Been Added!"))
            return redirect('stockquotes:add_stock')

    else:
        ticker = Stock.objects.all()
        output = []

        for ticker_item in ticker:

            api_request = requests.get(
                "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + str(ticker_item) + "&apikey"
                                                                                             "=AD3P4POXBTNNCS7D")
            try :
                api = json.loads(api_request.content)
                api_data = api.get('Global Quote', {})
                if not api_data :
                    processed_api = "Error..."
                else :
                    processed_api = {key.replace(' ', '_').replace('.', '') : value for key, value in api_data.items()}
                    output.append(processed_api)
            except Exception as e :
                processed_api = "Error..."
        return render(request, 'stockquotes/add_stock.html', {'ticker':ticker, 'output':output})

def delete(request, stock_id):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request, ("Stock Has Been Deleted!"))
    return redirect('stockquotes:add_stock')
# Create your views here.
