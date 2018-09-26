import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

df = pd.DataFrame(np.random.randn(5,3),index = list('abode'),columns = ['one','two','three'])
df.ix[1,:-1] = np.nan
df.ix[1:-1,2] = np.nan
print(df)

print(df.fillna(0))
df = df.apply(pd.value_counts)
print(df)