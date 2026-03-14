# Customer Segmentation
# Share-based clustering of delivery mode usage

# 1.create data

import pandas as pd
import numpy as np

np.random.seed(42)

lambdas = {
    'car': 18,
    'bike': 14,
    'box': 9,
    'van': 4,
    'van-floor': 2,
    'van-roof': 2,
    'truck': 0.6,
    'intercity': 0.4
}

n_users = 1000

data = {
    'user_id': np.arange(1, n_users + 1),
}

for col, lam in lambdas.items():
    data[col] = np.random.poisson(lam=lam, size=n_users)

df = pd.DataFrame(data)

# 2.view distribution

import matplotlib.pyplot as plt
features = (
    'car', 'box', 'bike', 'van', 'van-floor', 'van-roof', 'truck', 'intercity'
)
#df.to_csv('raw_df')

print(df.describe())
for col in features:
    plt.figure()
    plt.hist(df[col], bins=20)
    plt.title(col)
    plt.show()

# 3.feature engineering

df['total'] = df.iloc[:, 1:].sum(axis=1)
df = df[df['total'] > 0]
df_share = df.iloc[:,1:-1].div(df['total'], axis=0)


#df1['id']=df['user_id']
#df1 = df1[['id'] + [col for col in df1.columns if col != 'id']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_share_scaled = scaler.fit_transform(df_share)
df_share_scaled

# 4.finding k for k-means

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_values = range(2, 15)

# Elbow Method
elbow_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(df_share_scaled)
    elbow_scores.append(kmeans.inertia_)

# Silhouette Score
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(df_share_scaled)
    silhouette_avg = silhouette_score(df_share_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Visualization - Elbow Method
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(k_values, elbow_scores, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Visualization - Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# optimal k based on Elbow Method
optimal_k = k_values[np.argmin(elbow_scores)]
print(f"Optimal k (Elbow Method): {optimal_k}")

# optimal k based on Silhouette Score
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
print(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")


# 5.KMeans (FINAL MODEL)
from sklearn.cluster import KMeans
optimal_k = 10
kmeans = KMeans(
    n_clusters= optimal_k,
    random_state= 42,
    n_init= 20
)

df['cluster'] =  kmeans.fit_predict(df_share_scaled)
df

# 6.is this model good?

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(df_share_scaled, df['cluster'])
print(f"Silhouette Score (Final Model): {silhouette_avg}")

from sklearn.metrics import davies_bouldin_score
davies_bouldin = davies_bouldin_score(df_share_scaled, df['cluster'])
print(f"Davies-Bouldin Index: {davies_bouldin}")

from sklearn.metrics import calinski_harabasz_score
calinski_harabasz = calinski_harabasz_score(df_share_scaled, df['cluster'])
print(f"Calinski-Harabasz Index: {calinski_harabasz}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_share_scaled)

plt.figure(figsize=(8, 6))
for cluster in df['cluster'].unique():
    plt.scatter(df_pca[df['cluster'] == cluster, 0],
                df_pca[df['cluster'] == cluster, 1],
                label=f'Cluster {cluster}')
plt.title('PCA Scatter Plot of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# 7.Cluster Analysis 
# Mean share per cluster

features = [
    'car', 'box', 'bike', 'van', 'van-floor', 'van-roof', 'truck', 'intercity'
]

cluster_centers = (
    df
    .groupby('cluster')[features]
    .mean()
    .round(3)
)

print("\n Cluster Centers (Mean per Category) & cluster summery:")

cluster_count = df['cluster'].value_counts().sort_index()
cluster_share = (cluster_count/len(df)*100).round(2)
cluster_size = pd.DataFrame(
    {
        'user': cluster_count,
        'user_share' : cluster_share
    }
)

cluster_order_count = df.groupby('cluster')['total'].sum()
cluster_order_share = (cluster_order_count/cluster_order_count.sum()*100)

cluster_order = pd.DataFrame(
    {
        'total_order': cluster_order_count,
        'share_order' : cluster_order_share
    }
)

cluster_summary = ( 
    cluster_centers.merge(cluster_size,left_index=True, right_index=True ).merge(cluster_order,left_index=True, right_index=True)
)
print(cluster_summary)
#cluster_summary.to_csv('cluster_summary')



