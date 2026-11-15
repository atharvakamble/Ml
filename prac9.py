import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("student.csv")
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[["Age", "Marks"]]))

scores = [silhouette_score(X, KMeans(k, random_state=42, n_init=10).fit_predict(X)) for k in range(2, 5)]

best_k = range(2, 5)[scores.index(max(scores))]
labels = KMeans(best_k, random_state=42, n_init=10).fit_predict(X)

plt.plot(range(2, 5), scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

def print(arg1, best_k):
    ...

print("Optimal Clusters:", best_k)
print("Cluster Labels:", labels)
