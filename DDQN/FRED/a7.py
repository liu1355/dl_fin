import csv
import requests
import pandas as pd

FRED_BASE = 'https://www.quandl.com/api/v3/datasets/FRED/BASE/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_BASE)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    BASE_list = list(cr)
    for row in BASE_list:
        print(row)

BASE_list = pd.DataFrame(BASE_list)
BASE_list.to_csv('a7.csv', encoding = 'utf-8')
