import pandas as pd
import numpy as np

df1 = pd.DataFrame({'one': [1, 2, 3], 'two': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])

df2 = pd.DataFrame({'one': [1, 2, 3], 'two': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])

data = {'Item1': df1,'Item2': df2}

p = pd.panel(data, axis=1)

print(p)
