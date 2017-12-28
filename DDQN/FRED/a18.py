import csv
import requests
import pandas as pd

FRED_T5YIFR = 'https://www.quandl.com/api/v3/datasets/FRED/T5YIFR/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_T5YIFR)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    T5YIFR_list = list(cr)
    for row in T5YIFR_list:
        print(row)

T5YIFR_list = pd.DataFrame(T5YIFR_list)
T5YIFR_list.to_csv('a18.csv', encoding = 'utf-8')

