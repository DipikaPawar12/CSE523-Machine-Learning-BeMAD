import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'N':74, 'P':41, 'K':19, 'temperature':24, 'humidity':67.5, 'ph':6.58, 'rainfall':87.9298085})

print(r.json())
