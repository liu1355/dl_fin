import pandas as pd

r = range(1,29)

data = pd.DataFrame()
for i in r:
    path = "/path/DDQN/Economic Data/a%d.csv" % i
    frame = pd.read_csv(path, index_col = 0)

    if data.empty:
        data = frame
    else:
        data = data.merge(frame, on = '0', how = "outer")

print(data)
data.to_csv('merge.csv', encoding = 'utf-8')
