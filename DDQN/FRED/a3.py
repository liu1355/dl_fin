import csv
import requests
import pandas as pd

FRED_GDPPOT = 'https://www.quandl.com/api/v3/datasets/FRED/GDPPOT/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_GDPPOT)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    GDPPOT_list = list(cr)
    for row in GDPPOT_list:
        print(row)

GDPPOT_list = pd.DataFrame(GDPPOT_list)
GDPPOT_list.to_csv('a3.csv', encoding = 'utf-8')
