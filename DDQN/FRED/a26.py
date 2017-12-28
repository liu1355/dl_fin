import csv
import requests
import pandas as pd

FRED_UNEMPLOY = 'https://www.quandl.com/api/v3/datasets/UNEMPLOY/GDPDEF/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_UNEMPLOY)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    UNEMPLOY_list = list(cr)
    for row in UNEMPLOY_list:
        print(row)

UNEMPLOY_list = pd.DataFrame(UNEMPLOY_list)
UNEMPLOY_list.to_csv('a26.csv', encoding = 'utf-8')
