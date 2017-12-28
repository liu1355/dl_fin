import csv
import requests
import pandas as pd

FRED_EMRATIO = 'https://www.quandl.com/api/v3/datasets/FRED/EMRATIO/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_EMRATIO)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    EMRATIO_list = list(cr)
    for row in EMRATIO_list:
        print(row)

EMRATIO_list = pd.DataFrame(EMRATIO_list)
EMRATIO_list.to_csv('a25.csv', encoding = 'utf-8')
