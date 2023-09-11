from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import requests

api_key = '334c023697204e62b1cb9977f79888b2'
temp_img = "https://images.pexels.com/photos/3225524/pexels-photo-3225524.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=500"


def news_home ( request ) :
    page = request.GET.get('page', 1)
    category = request.GET.get('category', None)
    search = request.GET.get('search', None)

    if search :  # Add this block to handle search queries
        url = f"https://newsapi.org/v2/everything?q={search}&language=en&sortBy=publishedAt&apiKey={api_key}"

    elif category is None or category == "stockmarket":
        # get the top news
        url = f"https://newsapi.org/v2/everything?q=stock-market-news&sortBy=publishedAt&apiKey={api_key}"
    else :
        # get the search query request
        url = (
            "https://newsapi.org/v2/top-headlines?country={}&category={}&page={}&apiKey={}".format(
                "us", category, page, api_key
            ))
    r = requests.get(url=url)

    data = r.json()
    if data["status"] != "ok" :
        return HttpResponse("<h1>Request Failed</h1>")
    data = data["articles"]
    context = {
        "success" : True,
        "data" : [],
        "category" : category
    }
    # seprating the necessary data
    for i in data :
        context["data"].append({
            "title" : i["title"],
            "description" : "" if i["description"] is None else i["description"],
            "url" : i["url"],
            "image" : temp_img if i["urlToImage"] is None else i["urlToImage"],
            "publishedat" : i["publishedAt"]
        })

    # send the news feed to template in context
    return render(request, 'stocknews/news_home.html', context = context)
