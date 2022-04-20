import pandas as pd # useful for loading dataset
import matplotlib.pyplot as plt
import numpy as np # to perform array
from sklearn.cluster import KMeans

dataset = pd.read_csv(r'./CLV.csv')

income = dataset['INCOME'].values
spend = dataset['SPEND'].values
x = np.array(list(zip(income,spend)))


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

# plt.plot(range(1,11),wcss,color='red',marker='*')
# plt.title("K Means cluster")
# plt.xlabel("No. of cluster")
# plt.ylabel("WCSS")
# # plt.show()


model = KMeans(n_clusters=4,random_state=0)
y_mean = model.fit_predict(x)
# y_mean


sp = int(input("Enter your Income in k : "))
ot = int(input("Enter your Expenditure in k : "))
newinp = [[sp,ot]]
res = model.predict(newinp)
print("You are in ",res," Cluster")


plt.scatter(x[y_mean==0,0],x[y_mean==0,1],s=50,c='brown',label='1')
plt.scatter(x[y_mean==1,0],x[y_mean==1,1],s=50,c='blue',label='2')
plt.scatter(x[y_mean==2,0],x[y_mean==2,1],s=50,c='green',label='3')
plt.scatter(x[y_mean==3,0],x[y_mean==3,1],s=50,c='cyan',label='4')
plt.xlabel("Income")
plt.ylabel("Spend")
plt.legend()
plt.show()