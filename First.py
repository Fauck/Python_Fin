from fugle_marketdata import RestClient

client = RestClient(api_key = 'NjhkN2JjOTUtZDU0NC00MmFmLTkzMTEtOThlM2I1MTBiN2FkIDAzMTZlZWI4LTQ2YWEtNGUyOS05NGY5LTY2ZmUxMzA4MmYzNQ==')
stock = client.stock
x = stock.historical.candles(**{"symbol": "2330", "from": "2026-02-23", "to": "2026-02-24", "fields": "open,high,low,close"})
print(x)

