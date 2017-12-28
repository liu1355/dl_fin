import csv
import requests
import pandas as pd

FRED_CPILFESL = 'https://www.quandl.com/api/v3/datasets/FRED/CPILFESL/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_CPILFESL)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    CPILFESL_list = list(cr)
    for row in CPILFESL_list:
        print(row)

CPILFESL_list = pd.DataFrame(CPILFESL_list)
CPILFESL_list.to_csv('a5.csv', encoding = 'utf-8')