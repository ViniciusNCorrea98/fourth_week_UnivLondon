import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('em-9EhjTEemU7w7-EFnPcg_7aa34fc018d311e980c2cb6467517117_happyscore_income.csv')

data.sort_values('avg_income', inplace=True)

richest = data[data['avg_income'] > 15000]

happy = data['happyScore']
income = data['avg_income']


#To get the richest country with all informations
"""print(richest.iloc[-1])
print(richest.iloc[0:5])"""

rich_mean = np.mean(richest['avg_income'])
all_mean = np.mean(data['avg_income'])
ineq = data['income_inequality']

plt.xlabel('income')
plt.ylabel('happy score')
plt.scatter(income, happy, s=ineq*10, alpha=0.25)
plt.show()

print(all_mean, rich_mean)

for k, row in richest.iterrows():
    plt.text(row['avg_income'],
             row['happyScore'],
             row['country'])
    plt.scatter(row['avg_income'], row['happyScore'])

plt.show()

income_happy = np.column_stack((income, happy))
km_results = KMeans(n_clusters=3).fit(income_happy)

clusters = km_results.cluster_centers_

plt.scatter(income, happy)
plt.scatter(clusters[:,0], clusters[:,1], s=1000)
plt.show()


"""


print(happy)
print(income.max())
"""
