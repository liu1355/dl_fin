import csv
import requests
import pandas as pd

FRED_DGS30 = 'https://www.quandl.com/api/v3/datasets/FRED/DGS30/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_DGS30)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    DGS30_list = list(cr)
    for row in DGS30_list:
        print(row)

DGS30_list = pd.DataFrame(DGS30_list)
DGS30_list.to_csv('a15.csv', encoding = 'utf-8')
