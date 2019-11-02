import requests

response = requests.get(
    "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json")

print(response.status_code)
print(response.json())
