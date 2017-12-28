import csv
import requests
import pandas as pd

FRED_NROU = 'https://www.quandl.com/api/v3/datasets/FRED/NROU/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_NROU)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    NROU_list = list(cr)
    for row in NROU_list:
        print(row)

NROU_list = pd.DataFrame(NROU_list)
NROU_list.to_csv('a22.csv', encoding = 'utf-8')
