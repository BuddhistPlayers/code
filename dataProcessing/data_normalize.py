import pandas as pd
#input
df = pd.read_csv('C:\\Users\\subshine\\Desktop\\train1.csv', sep=',', header=0, index_col=0,encoding='ANSI')
                     #converters={column: json.loads for column in JSON_COLUMNS},
                     #dtype={'fullVisitorId': 'str'},
                     #nrows=None
#onehot
OneHot_COLUMNS = ["channelGrouping","device.browser","device.deviceCategory","device.operatingSystem","geoNetwork.city","geoNetwork.continent","geoNetwork.country","geoNetwork.networkDomain","geoNetwork.subContinent","trafficSource.medium","trafficSource.source"]
for column in OneHot_COLUMNS:
    c1=pd.get_dummies(df[column],prefix=column)
    df = df.drop(column, axis=1).merge(c1, right_index=True, left_index=True)

#Mapping Bool
bool_mapping = {"False": 0, "True": 1}
df['device.isMobile'] = df['device.isMobile'].map(bool_mapping)

#print(df['totals.hits'].value_counts())
#print(df['totals.newVisits'].value_counts())
#print(df['totals.pageviews'].value_counts())
#print(df['totals.transactionRevenue'].value_counts())
#print(df['totals.visits'].value_counts())


