import csv
import requests
import pandas as pd

FRED_M1 = 'https://www.quandl.com/api/v3/datasets/FRED/M1/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_M1)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    M1_list = list(cr)
    for row in M1_list:
        print(row)

M1_list = pd.DataFrame(M1_list)
M1_list.to_csv('a8.csv', encoding = 'utf-8')
