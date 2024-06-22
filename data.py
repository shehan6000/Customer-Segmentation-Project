import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic customer data
np.random.seed(42)
num_customers = 200

# Customer attributes
age = np.random.randint(18, 70, size=num_customers)
income = np.random.randint(20000, 150000, size=num_customers)
spending_score = np.random.randint(1, 100, size=num_customers)

# Create DataFrame
customer_data = pd.DataFrame({
    'CustomerID': range(1, num_customers + 1),
    'Age': age,
    'Annual Income (k$)': income,
    'Spending Score (1-100)': spending_score
})

# Save to CSV
customer_data.to_csv('customer_data.csv', index=False)

print(customer_data.head())
