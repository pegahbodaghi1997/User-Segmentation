# Customer Segmentation - Non-Business Users
# Share-based clustering of delivery mode usage

import clickhouse_connect
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------
# 1. Connect to Analytical Database
# -------------------------------------
client = clickhouse_connect.get_client(
    host='your_clickhouse_host',
    port=8123,
    user='your_username',
    password='your_password',
    database='analytics_db'
)

# -------------------------------------
# 2. Query: Delivery counts per user
# -------------------------------------
query = '''
SELECT
    user_id,
    countIf(delivery_mode = 'mode_bike')              AS bike,
    countIf(delivery_mode = 'mode_pooling')           AS pooling,
    countIf(delivery_mode = 'mode_bike_boxless')      AS bike_boxless,
    countIf(delivery_mode = 'mode_van')               AS van,
    countIf(delivery_mode = 'mode_van_roof')          AS van_roof,
    countIf(delivery_mode = 'mode_van_floor')         AS van_floor,
    countIf(delivery_mode = 'mode_van_heavy')         AS van_heavy,
    countIf(delivery_mode = 'mode_truck')             AS truck,
    countIf(delivery_mode = 'mode_home_moving')       AS home_moving,
    countIf(delivery_mode ILIKE '%car_box%')          AS car_box
FROM delivery_orders_fact
WHERE order_date BETWEEN '2024-10-30' AND '2025-10-30'
  AND user_id NOT IN (
      SELECT DISTINCT user_id
      FROM business_users_dim
  )
GROUP BY user_id
'''

result = client.query(query)
df = pd.DataFrame(result.result_rows, columns=result.column_names)

print(df.head())
print(f"Total users: {len(df)}")

# -------------------------------------
# 3. Feature Preparation
# -------------------------------------
delivery_features = [
    'bike', 'pooling', 'bike_boxless', 'van',
    'van_roof', 'van_floor', 'van_heavy',
    'truck', 'home_moving', 'car_box'
]

X = df[delivery_features]

# -------------------------------------
# 4. Feature Engineering: Share per mode
# -------------------------------------
df['total_orders'] = X.sum(axis=1)

# Remove users with zero orders (safety check)
df = df[df['total_orders'] > 0].copy()

for col in delivery_features:
    df[f"{col}_share"] = df[col] / df['total_orders']

share_features = [f"{col}_share" for col in delivery_features]
X_share = df[share_features]

# Standardize to avoid scale bias
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_share)

# -------------------------------------
# 5. Determine Optimal Number of Clusters
# -------------------------------------
sse = []
K_range = range(2, 10)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    sse.append(model.inertia_)

plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

# -------------------------------------
# 6. Fit Final K-Means Model
# -------------------------------------
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------------------
# 7. Cluster Analysis
# -------------------------------------
cluster_centers = (
    df.groupby('cluster')[share_features]
      .mean()
      .round(3)
)

print("\nCluster Centers (Average Usage Share):")
print(cluster_centers)

# -------------------------------------
# 8. Cluster Size & Distribution
# -------------------------------------
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(df) * 100).round(2)

cluster_stats = pd.DataFrame({
    'users': cluster_counts,
    'user_percentage': cluster_percentages
})

print("\nCluster Distribution:")
print(cluster_stats)

# -------------------------------------
# 9. Order Share per Cluster
# -------------------------------------
orders_per_cluster = df.groupby('cluster')['total_orders'].sum()
total_orders = orders_per_cluster.sum()

order_share = (orders_per_cluster / total_orders * 100).round(2)

order_stats = pd.DataFrame({
    'total_orders': orders_per_cluster,
    'order_share_percentage': order_share
})

print("\nOrder Share by Cluster:")
print(order_stats)

# -------------------------------------
# 10. Final Cluster Summary
# -------------------------------------
final_summary = (
    cluster_centers
    .merge(cluster_stats, left_index=True, right_index=True)
    .merge(order_stats, left_index=True, right_index=True)
)

print("\nFinal Cluster Summary:")
print(final_summary)

# -------------------------------------
# 11. Save Results
# -------------------------------------
df.to_csv('user_delivery_segments.csv', index=False)
final_summary.to_csv('cluster_summary.csv', index=False)
