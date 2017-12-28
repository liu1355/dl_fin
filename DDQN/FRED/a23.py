import csv
import requests
import pandas as pd

FRED_NROUST = 'https://www.quandl.com/api/v3/datasets/FRED/NROUST/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_NROUST)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    NROUST_list = list(cr)
    for row in NROUST_list:
        print(row)

NROUST_list = pd.DataFrame(NROUST_list)
NROUST_list.to_csv('a23.csv', encoding = 'utf-8')
