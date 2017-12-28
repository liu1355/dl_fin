import csv
import requests
import pandas as pd

FRED_GDPC1 = 'https://www.quandl.com/api/v3/datasets/FRED/GDPC1/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_GDPC1)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    GDPC1_list = list(cr)
    for row in GDPC1_list:
        print(row)

GDPC1_list = pd.DataFrame(GDPC1_list)
GDPC1_list.to_csv('a2.csv', encoding = 'utf-8')
