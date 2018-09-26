import pandas as pd
#input
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\train1.csv',
                     #converters={column: json.loads for column in JSON_COLUMNS},
                     #dtype={'fullVisitorId': 'str'},
                     nrows=None)