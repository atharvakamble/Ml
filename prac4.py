import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv("student.csv")

df = df.fillna(df.mode().iloc[0])

le = LabelEncoder()
for col in ["Name", "Gender", "City"]:
    df[col] = le.fit_transform(df[col])

X = df[["Age", "Marks", "Gender", "City"]]

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_reduced.shape)
print("Variance Retained:", pca.explained_variance_ratio_.sum())

pd.DataFrame(X_reduced, columns=["PC1", "PC2"]).to_csv(
    "student_pca.csv", index=False
)

print("PCA completed successfully!")
