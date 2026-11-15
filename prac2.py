import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load student dataset
df = pd.read_csv("student.csv")

print("Original Dataset:")
print(df)

# Handle missing numerical values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

# Handle missing categorical values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["City"] = df["City"].fillna(df["City"].mode()[0])

print("\nDataset After Handling Missing Values:")
print(df)

# Scale numerical attributes
scaler = StandardScaler()
df[["Age", "Marks"]] = scaler.fit_transform(df[["Age", "Marks"]])

print("\nScaled Dataset:")
print(df)

# Save final dataset
df.to_csv("model_ready_student.csv", index=False)

print("\nModel-ready dataset saved successfully!")
