import csv
import requests
import pandas as pd

FRED_M2 = 'https://www.quandl.com/api/v3/datasets/FRED/M2/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_M2)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    M2_list = list(cr)
    for row in M2_list:
        print(row)

M2_list = pd.DataFrame(M2_list)
M2_list.to_csv('a9.csv', encoding = 'utf-8')
