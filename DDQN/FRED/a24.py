import csv
import requests
import pandas as pd

FRED_CIVPART = 'https://www.quandl.com/api/v3/datasets/FRED/CIVPART/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_CIVPART)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    CIVPART_list = list(cr)
    for row in CIVPART_list:
        print(row)

CIVPART_list = pd.DataFrame(CIVPART_list)
CIVPART_list.to_csv('a24.csv', encoding = 'utf-8')
