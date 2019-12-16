import requests
import json

class TwitterApi:
    def __init__(self, API):
        self.API = API
        self.response = None
        self.status = None
        self.unformattedResponse = None
        self.formattedResponse = None

    def request(self, link):
        self.response = requests.get(link, auth=("Team_Stonks", "ucfai2019"))
        self.status = self.response.status_code
        self.unformattedResponse = self.response.json()
        self.formattedResponse = json.dumps(self.unformattedResponse, sort_keys=True, indent=4)
    
    def details(self, *headers):
        headerResults = ""
        for x in headers:
            headerResults += (x + ": " + str(self.response.headers[x]) + " || ")
        return  headerResults + "Status: " + str(self.status)

stockTwit = TwitterApi("StockTwits")

#stockTwit.request("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json")

print(stockTwit.API)