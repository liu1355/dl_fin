import csv
import requests
import pandas as pd

FRED_TEDRATE = 'https://www.quandl.com/api/v3/datasets/FRED/TEDRATE/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_TEDRATE)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    TEDRATE_list = list(cr)
    for row in TEDRATE_list:
        print(row)

TEDRATE_list = pd.DataFrame(TEDRATE_list)
TEDRATE_list.to_csv('a19.csv', encoding = 'utf-8')
