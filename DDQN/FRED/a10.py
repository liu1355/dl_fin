import csv
import requests
import pandas as pd

FRED_M1V = 'https://www.quandl.com/api/v3/datasets/FRED/M1V/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_M1V)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    M1V_list = list(cr)
    for row in M1V_list:
        print(row)

M1V_list = pd.DataFrame(M1V_list)
M1V_list.to_csv('a10.csv', encoding = 'utf-8')
