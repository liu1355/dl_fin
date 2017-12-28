import csv
import requests
import pandas as pd

FRED_CPIAUCSL = 'https://www.quandl.com/api/v3/datasets/FRED/CPIAUCSL/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_CPIAUCSL)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    CPIAUCSL_list = list(cr)
    for row in CPIAUCSL_list:
        print(row)

CPIAUCSL_list = pd.DataFrame(CPIAUCSL_list)
CPIAUCSL_list.to_csv('a4.csv', encoding = 'utf-8')