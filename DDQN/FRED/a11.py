import csv
import requests
import pandas as pd

FRED_M2V = 'https://www.quandl.com/api/v3/datasets/FRED/M2V/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_M2V)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    M2V_list = list(cr)
    for row in M2V_list:
        print(row)

M2V_list = pd.DataFrame(M2V_list)
M2V_list.to_csv('a11.csv', encoding = 'utf-8')