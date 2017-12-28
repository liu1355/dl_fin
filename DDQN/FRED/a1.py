import csv
import requests
import pandas as pd

FRED_GDP = 'https://www.quandl.com/api/v3/datasets/FRED/GDP/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_GDP)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    GDP_list = list(cr)
    for row in GDP_list:
        print(row)

GDP_list = pd.DataFrame(GDP_list)
GDP_list.to_csv('a1.csv', encoding = 'utf-8')
