import csv
import requests
import pandas as pd

FRED_DFF = 'https://www.quandl.com/api/v3/datasets/FRED/DFF/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_DFF)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    DFF_list = list(cr)
    for row in DFF_list:
        print(row)

DFF_list = pd.DataFrame(DFF_list)
DFF_list.to_csv('a12.csv', encoding = 'utf-8')

