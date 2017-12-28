import csv
import requests
import pandas as pd

FRED_DTB3 = 'https://www.quandl.com/api/v3/datasets/FRED/DTB3/data.csv?api_key=6CbgFEPrywyyFy1yNywC'

with requests.Session() as s:
    download = s.get(FRED_DTB3)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter = ',')
    DTB3_list = list(cr)
    for row in DTB3_list:
        print(row)

DTB3_list = pd.DataFrame(DTB3_list)
DTB3_list.to_csv('a13.csv', encoding = 'utf-8')
