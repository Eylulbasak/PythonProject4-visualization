import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as pt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr,\
    kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

populasyon = np.random.randint(0, 80, 10000)
np.random.seed(115)
#orneklem
sampling = np.random.choice(a=populasyon, size=100)

#descriptive analysis
df= sns.load_dataset('tips')

sms.DescrStatsW(df['total_bill']).tconfint_mean()

df['total_bill']= df['total_bill'] - df['tip']
df.plot.scatter('total_bill', 'tip')
plt.show()
df['tip'].corr(df['total_bill'])