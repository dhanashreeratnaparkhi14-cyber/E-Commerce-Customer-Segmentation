import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load customer dataset
data = pd.read_csv("dataset/Mall_Customers.csv")

# Display first five rows
print(data.head())

# Display dataset information
print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Select useful columns for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Customer Data:")
print(X.head())

print("\nSelected Data Shape:")
print(X.shape)

# Find optimal number of clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Display Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Create customer clusters using K-Means

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Train the model
kmeans.fit(X)

# Add cluster labels to the dataset
data["Cluster"] = kmeans.labels_

# Display customer clusters
print("\nCustomer Clusters:")
print(data.head(10))
# Visualize customer clusters

plt.figure(figsize=(8, 5))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=data["Cluster"],
    cmap="rainbow"
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
# Check the shape of dataset
print("Dataset Shape:")
print(data.shape)

# Check column names
print("\nColumn Names:")
print(data.columns)

# Check dataset information
print("\nDataset Information:")
data.info()

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Analyze Customer Groups

print("\nCustomer Segment Analysis:")

segment_analysis = data.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(segment_analysis)
# Give Names to Customer Segments

cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Budget High Spenders",
    3: "Careful Customers",
    4: "Budget Customers"
}

data["Customer Segment"] = data["Cluster"].map(cluster_names)

print("\nFinal Customer Segmentation:")
print(data[["CustomerID", "Annual Income (k$)",
            "Spending Score (1-100)",
            "Customer Segment"]].head(10))
# Save Final Customer Segmentation Dataset

data.to_csv("Customer_Segmentation_Result.csv", index=False)

print("\nFinal dataset saved successfully!")