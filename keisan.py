import pandas as pd
import numpy as np
df = pd.read_csv('mnist/result/data21_val.csv', header=None, encoding = 'shift_jis')
df = df.iloc[:,2]
print('平均', df.mean())
print('分散', np.var(df))