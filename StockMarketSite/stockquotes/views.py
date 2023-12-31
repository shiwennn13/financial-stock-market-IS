from django.shortcuts import render, redirect
from .models import Stock
from .forms import StockForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required


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


@login_required(login_url='login')
def add_stock ( request ) :
    import requests
    import json

    def validate_stock_symbol ( stock_symbol ) :
        api_key = "0BA8N0PMQV2APZGB"
        api_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_symbol}&apikey={api_key}"
        api_request = requests.get(api_url)
        api_data = json.loads(api_request.content).get('Global Quote', {})
        return bool(api_data)

    if request.method == 'POST' :
        form = StockForm (request.POST or None)

        if form.is_valid():
            new_stock = form.save(commit=False)
            new_stock.user = request.user  # Associate this stock with the logged-in user
            stock_symbol = new_stock.ticker

            # Check if the stock symbol already exists in the user's watchlist
            existing_stock = Stock.objects.filter(user=request.user, ticker=stock_symbol).first()

            if existing_stock :
                messages.error(request, "This stock symbol already exists in your watchlist.")
            else:
                is_valid_stock = validate_stock_symbol(stock_symbol)
                if is_valid_stock:
                    new_stock.save()
                    messages.success(request, "Stock Has Been Added!")
                    return redirect('stockquotes:add_stock')
                else:
                    messages.error(request, "Invalid stock symbol. Please try again.")
        else :
            messages.error(request, "There was a problem adding the stock. Please try again.")
        return redirect('stockquotes:add_stock')

    else:
        ticker = Stock.objects.filter(user=request.user)# Filter stocks by the logged-in user
        output = []
        overview_output = []

        for ticker_item in ticker:
            # Global Quote API request
            api_request = requests.get(
                "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + str(ticker_item) + "&apikey=0BA8N0PMQV2APZGB")
            # Overview API request
            overview_api_request = requests.get(
                "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + str(ticker_item) + "&apikey=0BA8N0PMQV2APZGB")
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