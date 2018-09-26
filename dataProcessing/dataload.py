import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
DROP_COLUMNS = ['date','visitStartTime','device.browserVersion','device.browserSize','device.operatingSystemVersion','device.mobileDeviceBranding','device.mobileDeviceModel','device.mobileInputSelector','device.mobileDeviceInfo','device.mobileDeviceMarketingName','device.flashVersion','device.language','device.screenColors','device.screenResolution','geoNetwork.region','geoNetwork.metro','geoNetwork.cityId','geoNetwork.latitude','geoNetwork.longitude','geoNetwork.networkLocation','socialEngagementType','totals.bounces','trafficSource.campaign','trafficSource.keyword','trafficSource.isTrueDirect','trafficSource.adwordsClickInfo.criteriaParameters','visitId','fullVisitorId','sessionId','trafficSource.adContent','trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.isVideoAd','trafficSource.adwordsClickInfo.page','trafficSource.adwordsClickInfo.slot','trafficSource.campaignCode','trafficSource.referralPath']
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\train.csv',
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},
                     nrows=None)

for column in JSON_COLUMNS:
    c1 = np.array(df[column])
    c2 = c1.tolist()
    column_as_df = json_normalize(c2)
    column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

'''
print ("缺失值数量")
print(df.shape[0]- df.count())  #统计缺失值数量
print ("缺失值比例")
print(1 - df.count()/df.shape[0]) #统计缺失值比例
'''
df = df.drop(DROP_COLUMNS, axis=1)
df = df.fillna(0.0)
df.to_csv('C:\\Users\\DELL\\Desktop\\train1.csv', sep=',', header=True, index=True)
#print(df)
#df1 = df.apply(pd.value_counts)
#df1.to_csv('C:\\Users\\DELL\\Desktop\\count.csv', sep=',', header=True, index=True)