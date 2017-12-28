import os

r = range(1,29)
for i in r:
    file = 'python a%d.py' % i
    os.system(file)

os.system('python merge.py')