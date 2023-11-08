import pandas as pd
import numpy as np
import statsmodels.api as sm
import openpyxl

from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial import cKDTree

# Total Market Cap and the number of transactions in the Ethereum network were taken as regressor variables.
# Commissions are excluded because they do not have a significant explanatory effect.

tmc = 'totalmarketcap.csv'
txs = 'transactions.csv'
regs = 'registrations.csv'

df1 = pd.read_csv(tmc, usecols=[1], skiprows=[1])
df2 = pd.read_csv(txs, usecols=[1], skiprows=[1])
merged_df = pd.concat([df1, df2], axis=1)
regs_df = pd.read_csv(regs)

# Due to the multicollinearity of tmc and txs variables, combining them improves model results and aids in interpretation. 
# The format for combining variables is the principal component method for 1 component.

pca = PCA(n_components=1)
pca_result = pca.fit_transform(merged_df)
exog = pd.DataFrame(pca_result, columns=['PCA1'])

# There are a large number of outliers in the registration schedule. 
# Detection method: Fisher's z-test. 
# The imputation format is k-NN. 
# The meaning of the variables is empirically determined, and changes require expert judgment.

z_scores = stats.zscore(regs_df['Value'])

critical_z = 2.5
iterations = 5
neighbors = 7

kdtree = cKDTree(regs_df['Value'].values.reshape(-1, 1))
outliers = regs_df[abs(z_scores) > critical_z]

for _ in range(iterations):
    for index in outliers.index:
        _, neighbor_indices = kdtree.query([regs_df.at[index, 'Value']], k=neighbors)
        neighbor_values = [regs_df.at[i, 'Value'] for i in neighbor_indices]
        imputed_value = np.mean(neighbor_values)
        regs_df.at[index, 'Value'] = imputed_value

# The regression is performed based on a 37-week data shift. 
# To select the optimal shift, it is necessary not only to identify the highest correlation indicators, but also to conduct a visual analysis of the graph, since the correlation of dynamics may not have an economic justification, even if there is a high r indicator.

shift = 37

Y = regs_df['Value'].iloc[shift:].reset_index(drop=True)
X = exog.iloc[:-(shift)+1, :].reset_index(drop=True)
X = sm.add_constant(X)

linear_model = sm.OLS(Y, X).fit()
linear_r_squared = linear_model.rsquared

exponential_model = sm.OLS(np.log(Y), X).fit()
exponential_r_squared = exponential_model.rsquared

# To determine the most accurate functional relationship, the coefficient of determination of linear and exponential regression is compared.
# It was experimentally found that only exponential or linear regression produces any significant results. If suddenly the dependence function changes, for example, to a power or logarithmic one, then the model should be reconsidered as a whole, and not just transform the variables.

best_model = None
if linear_r_squared >= exponential_r_squared:
    best_model = 'linear'
    Y_pred = linear_model.predict(X)
else:
    best_model = 'exponential'
    Y_pred = exponential_model.predict(X)
    Y_pred = np.exp(Y_pred)
    Y = np.log(Y)

errors_model1 = np.abs((np.exp(Y)) - Y_pred)
mean_error_model1 = np.mean(errors_model1)
print(f'Mean Absolute Error for model 1: {mean_error_model1}')

Y_lagged = Y.shift(1)
X_comb = pd.concat([X, Y_lagged], axis=1)

model2 = sm.OLS(Y[1:], X_comb[1:]).fit()
Y_pred2 = model2.predict(X_comb)[1:]

if best_model == 'linear':
    errors_model2 = abs(Y - Y_pred2)
else:
    Y_pred2 = np.exp(Y_pred2)
    errors_model2 = np.exp(Y) - Y_pred2

mean_error_model2 = np.mean(errors_model2)
print(f'Mean Absolute Error for model 2: {mean_error_model2}')

num_observations = len(Y_pred)
confidence_level = 100
confidence_level -= (num_observations // 30)
confidence_level = max(confidence_level, 0)

real_values = regs_df['Value'].iloc[shift+1:].reset_index(drop=True)

# The value of the influence of external factors is determined by the accuracy ratio of the first model.
# The threshold is determined based on the number of observations.

exog_inf = [(min(value, confidence_level)) for value in (Y_pred / real_values) * 100]
exog_inf = [None] * (shift+1) + exog_inf
exog_df = pd.concat([regs_df['Date'], pd.Series(exog_inf, name='External_Influence')], axis=1).dropna(subset=['External_Influence'])

# The value of the influence marketing is determined by the mean percentage error of the second model.

marketing_inf = []

for i in range(1,len(errors_model2)):
    error = errors_model2.iloc[i]
    observed_value = np.exp(Y.iloc[i])
    
    if error < 0:
        marketing_inf.append(0)
    else:
        effectiveness = (observed_value - error) / observed_value
        marketing_inf.append(effectiveness)

marketing_inf = ((100 - exog_df['External_Influence']) * marketing_inf)

# The efficiency of autoregressive factors is the remainder.

ar_inf = 100 - (exog_df['External_Influence'] + marketing_inf)

results = pd.DataFrame({
    'Date': regs_df['Date'].iloc[shift+1:],
    'External_Influence': round(pd.Series(exog_inf, name='External_Influence'), 2),
    'Marketing': round(pd.Series(marketing_inf, name='Marketing'), 2),
    'AR/other': round(pd.Series(ar_inf, name='AR/other'), 2)
}).dropna(subset=['External_Influence'])

results.to_csv('results.csv', index=False)
results.to_excel('relults.xlsx', index=False)