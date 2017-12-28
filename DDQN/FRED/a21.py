import csv
import requests
import pandas as pd

FRED_UNRATE = 'https://www.quandl.com/api/v3/datasets/FRED/UNRATE/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_UNRATE)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    UNRATE_list = list(cr)
    for row in UNRATE_list:
        print(row)

UNRATE_list = pd.DataFrame(UNRATE_list)
UNRATE_list.to_csv('a21.csv', encoding = 'utf-8')
