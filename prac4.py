import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("student.csv")

# Handle missing values
df = df.fillna(df.mode().iloc[0])

# Encode categorical variables
le = LabelEncoder()
for col in ["Name", "Gender", "City"]:
    df[col] = le.fit_transform(df[col])

# Select features
X = df[["Age", "Marks", "Gender", "City"]]

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Display results
print("Original Shape:", X.shape)
print("Reduced Shape:", X_reduced.shape)
print("Variance Retained:", pca.explained_variance_ratio_.sum())

# Save reduced dataset
pd.DataFrame(X_reduced, columns=["PC1", "PC2"]).to_csv(
    "student_pca.csv", index=False
)

print("PCA completed successfully!")
