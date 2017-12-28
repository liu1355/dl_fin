import csv
import requests
import pandas as pd

FRED_T10YIE = 'https://www.quandl.com/api/v3/datasets/FRED/T10YIE/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_T10YIE)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    T10YIE_list = list(cr)
    for row in T10YIE_list:
        print(row)

T10YIE_list = pd.DataFrame(T10YIE_list)
T10YIE_list.to_csv('a17.csv', encoding = 'utf-8')
