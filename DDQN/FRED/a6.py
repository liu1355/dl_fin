import csv
import requests
import pandas as pd

FRED_GDPDEF = 'https://www.quandl.com/api/v3/datasets/FRED/GDPDEF/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_GDPDEF)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    GDPDEF_list = list(cr)
    for row in GDPDEF_list:
        print(row)

GDPDEF_list = pd.DataFrame(GDPDEF_list)
GDPDEF_list.to_csv('a6.csv', encoding = 'utf-8')