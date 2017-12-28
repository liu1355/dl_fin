import csv
import requests
import pandas as pd

FRED_DGS10 = 'https://www.quandl.com/api/v3/datasets/FRED/DGS10/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_DGS10)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    DGS10_list = list(cr)
    for row in DGS10_list:
        print(row)

DGS10_list = pd.DataFrame(DGS10_list)
DGS10_list.to_csv('a14.csv', encoding = 'utf-8')
