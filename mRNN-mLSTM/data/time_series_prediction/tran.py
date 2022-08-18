import pandas as pd
import numpy as np
p = '../data/time_series_prediction/DJI.csv'

d = pd.read_csv(p)
d['x'] = d['x'] * 100

pd.set_option('display.precision', 7)

d.to_csv(p, index=False, float_format='%.7f')
