# Load the data
customer_data = pd.read_csv('customer_data.csv')

# Preprocessing: Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(customer_data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means with the optimal number of clusters (let's assume it's 5 from the Elbow Method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Save the clustered data to a CSV file
customer_data.to_csv('customer_segmented_data.csv', index=False)

print(customer_data.head())
