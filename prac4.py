
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv("student.csv").ffill()

le = LabelEncoder()
for col in ["Name", "Gender", "City"]:
    df[col] = le.fit_transform(df[col])

X = StandardScaler().fit_transform(df[["Age", "Marks", "Gender", "City"]])

pca = PCA(n_components=2)
result = pca.fit_transform(X)

print("Original Shape:", X.shape)
print("Reduced Shape:", result.shape)
print("Variance Retained:", pca.explained_variance_ratio_.sum())

pd.DataFrame(result, columns=["PC1", "PC2"]).to_csv("student_pca.csv", index=False)

print("PCA completed!")
