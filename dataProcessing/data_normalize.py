import pandas as pd
#input
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\train1.csv', sep=',', header=0, index_col=0,encoding='ANSI')
                     #converters={column: json.loads for column in JSON_COLUMNS},
                     #dtype={'fullVisitorId': 'str'},
                     #nrows=None
print(df)