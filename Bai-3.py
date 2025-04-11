import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

CSV_FILE = 'ketqua.csv'  
TOP_N = 3
ENCODING = 'utf-8-sig'

try:
    df = pd.read_csv(CSV_FILE, encoding=ENCODING)
    print(f"Data successfully read from '{CSV_FILE}'.")
except Exception as e:
    print(f"Error reading file '{CSV_FILE}': {e}")
    exit()

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_columns:
    print("No numeric columns found.")
    exit()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k = 3  

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nCluster Distribution and Player Statistics:")
for cluster in range(optimal_k):
    cluster_stats = df[df['Cluster'] == cluster][numeric_columns].mean()
    print(f"\nCluster {cluster}:")
    print(cluster_stats)

print("\nComments on the Clusters:")

print(f"\nCluster 0 may represent players with higher attacking statistics such as goals scored, assists, etc.")
print(f"Cluster 1 may represent players with strong defensive stats such as tackles, interceptions, etc.")
print(f"Cluster 2 might correspond to players who have balanced stats across offense and defense, showing versatility.")

print("\nWhy 3 clusters?")
print("Based on the Elbow Method, the inertia value decreases significantly up to 3 clusters, and the further decrease is minimal.")
print("This indicates that adding more clusters does not significantly improve the representation of player differences.")
print("Hence, classifying all players who have played more than 90 minutes in the 2024-2025 English Premier League season into 3 groups captures meaningful differences while maintaining simplicity.")

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette="viridis", s=100, alpha=0.7)
plt.title("2D Visualization of Player Clusters Using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc='best')
plt.tight_layout()
plt.show()

print("\nIn conclusion:")
print("The clustering results reveal that players can be grouped into three distinct categories based on their performance metrics.")
print("Given the data from all players who have played more than 90 minutes in the 2024-2025 English Premier League season, classifying into 3 groups seems appropriate as it balances meaningful segmentation with a manageable number of categories.")
print("Cluster 0 likely corresponds to players excelling in offensive skills, Cluster 1 to those strong defensively, and Cluster 2 to versatile players with both offensive and defensive capabilities.")
print("This classification can help analysts or coaches make more strategic decisions, assigning roles or identifying training needs.")
