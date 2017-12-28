import csv
import requests
import pandas as pd

FRED_DPRIME = 'https://www.quandl.com/api/v3/datasets/FRED/DPRIME/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_DPRIME)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    DPRIME_list = list(cr)
    for row in DPRIME_list:
        print(row)

DPRIME_list = pd.DataFrame(DPRIME_list)
DPRIME_list.to_csv('a20.csv', encoding = 'utf-8')
