import requests, os

url = "http://127.0.0.1:5000/"
os.environ['NO_PROXY'] = '127.0.0.1'

query = 'Can you please show me flights from new york to los angeles arriving before 6 pm'

r = requests.get('http://127.0.0.1:5000/flightreservation/api/get_entities/{0}'.format(query))
print(r.text)