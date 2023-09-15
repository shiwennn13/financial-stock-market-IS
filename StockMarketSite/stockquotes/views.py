from django.shortcuts import render, redirect
from .models import Stock
from .forms import StockForm
from django.contrib import messages



def qhome ( request ) :

    ticker_symbol = "NASDAQ:NDX"  # Default ticker symbol

    if request.method == 'POST' :
        ticker = request.POST['ticker']  # 'ticker' is the one in the qhome.html <input> name
        ticker_symbol = ticker

    return render(request, 'stockquotes/qhome.html', {'ticker_symbol' : ticker_symbol})

    # import requests
    # import json
    #
    # processed_api = {}  # Initialize as an empty dictionary
    # ticker_symbol = "AAPL"
    #
    #
    # if request.method == 'POST' :
    #     ticker = request.POST['ticker']  # 'ticker' is the one in the qhome.html <input> name
    #     # apikey: AD3P4POXBTNNCS7D
    #     ticker_symbol = ticker
    #
    #     # Global Quote API request
    #     api_request = requests.get(
    #         "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + ticker + "&apikey"
    #                                                                                      "=AD3P4POXBTNNCS7D")
    #     # Overview API request
    #     overview_api_request = requests.get(
    #         "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + ticker + "&apikey=AD3P4POXBTNNCS7D")
    #
    #
    #     try :
    #         api = json.loads(api_request.content)
    #         overview_api = json.loads(overview_api_request.content)
    #         print("Global Quote API:", api)
    #         print("Overview API:", overview_api)
    #         api_data = api.get('Global Quote', {})
    #         if not api_data :
    #             processed_api = "Error..."
    #         else :
    #             processed_api = {key.replace(' ', '_').replace('.', '') : value for key, value in api_data.items()}
    #     except Exception as e :
    #         print("Exception:", e)
    #         processed_api = "Error..."
    #         overview_api = "Error in Company Overview"
    #
    #     return render(request, 'stockquotes/qhome.html', {'api': processed_api, 'overview': overview_api, 'ticker_symbol': ticker_symbol})
    # else :
    #     return render(request, 'stockquotes/qhome.html', {'ticker' : "Enter a ticker symbol above"})

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
        overview_output = []

        for ticker_item in ticker:
            # Global Quote API request
            api_request = requests.get(
                "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + str(ticker_item) + "&apikey=AD3P4POXBTNNCS7D")
            # Overview API request
            overview_api_request = requests.get(
                "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + str(ticker_item) + "&apikey=AD3P4POXBTNNCS7D")
            try :
                api = json.loads(api_request.content)
                overview_api = json.loads(overview_api_request.content)
                print("Global Quote API:", api)
                print("Overview API:", overview_api)
                api_data = api.get('Global Quote', {})
                if not api_data :
                    processed_api = "Error..."
                else :
                    processed_api = {key.replace(' ', '_').replace('.', '') : value for key, value in api_data.items()}
                    output.append(processed_api)
                    overview_output.append(overview_api)
            except Exception as e :
                print("Exception:", e)
                processed_api = "Error..."
                overview_api = "Error in Company Overview"


            # Zip the two lists together
        output_and_overview = zip(output, overview_output)
        return render(request, 'stockquotes/add_stock.html',
                      {'ticker': ticker, 'output_and_overview': output_and_overview})

def delete(request, stock_id):
    item = Stock.objects.get(pk=stock_id)
    item.delete()
    messages.success(request, ("Stock Has Been Deleted!"))
    return redirect('stockquotes:add_stock')
# Create your views here.