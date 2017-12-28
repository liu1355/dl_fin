import csv
import requests
import pandas as pd

FRED_MANEMP = 'https://www.quandl.com/api/v3/datasets/FRED/MANEMP/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_MANEMP)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    MANEMP_list = list(cr)
    for row in MANEMP_list:
        print(row)

MANEMP_list = pd.DataFrame(MANEMP_list)
MANEMP_list.to_csv('a28.csv', encoding = 'utf-8')
