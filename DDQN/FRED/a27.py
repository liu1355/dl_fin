import csv
import requests
import pandas as pd

FRED_PAYEMS = 'https://www.quandl.com/api/v3/datasets/FRED/PAYEMS/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_PAYEMS)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    PAYEMS_list = list(cr)
    for row in PAYEMS_list:
        print(row)

PAYEMS_list = pd.DataFrame(PAYEMS_list)
PAYEMS_list.to_csv('a27.csv', encoding = 'utf-8')