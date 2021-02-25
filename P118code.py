import pandas as pd
import plotly.express as px
import plotly.graph_objects as gp
import csv
import plotly.figure_factory as pf
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sb
from sklearn.cluster import KMeans

df = pd.read_csv("P118.csv")


petal_size_list = df["Size"].tolist()

sepal_size_list = df["Light"].tolist()

fig = px.scatter(df, x = "Size", y= "Light")

#fig.show()


#__________________________________________________________________



X = df.iloc[:, [0,1]].values
print(X)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 30)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


y_kmeans = kmeans.fit_predict(X)




plt.figure(figsize=(15,7))
sb.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'black', label = 'Cluster 1')
sb.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sb.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'red', label = 'Cluster 3')
sb.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'green', label = 'Cluster 4')
sb.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'yellow', label = 'Cluster 5')
plt.legend()
