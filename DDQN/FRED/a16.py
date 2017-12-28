import csv
import requests
import pandas as pd

FRED_T5YIE = 'https://www.quandl.com/api/v3/datasets/FRED/T5YIE/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_T5YIE)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    T5YIE_list = list(cr)
    for row in T5YIE_list:
        print(row)

T5YIE_list = pd.DataFrame(T5YIE_list)
T5YIE_list.to_csv('a16.csv', encoding = 'utf-8')
