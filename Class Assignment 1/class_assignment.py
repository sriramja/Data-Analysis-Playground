import pandas as pd

df = pd.read_csv('/content/database.csv', header=None) 
header = list(df.iloc[0])
header

df_new = pd.read_csv('/content/database.csv', header=1) 
df_new.to_csv("database_modified.csv")

"""Part -2 """

df = pd.read_csv('/content/database.csv') 

df1 = df.iloc[:round(len(df)/2),:]
df2 = df.iloc[len(df)-round(len(df)/2):,:]
df2.loc[0] = header


df1.to_csv("database1.csv")
df2.to_csv("database2.csv")

"""Part 3"""

df1.to_json(r'database1.json')
df2.to_json(r'database2.json')

"""Part - 4

"""

import json
import csv

with open('/content/EmployeeData.json') as json_file:
    jsondata = json.load(json_file)
 
data_file = open('EmployeeData.csv', 'w', newline='')
csv_writer = csv.writer(data_file)
 
count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()

