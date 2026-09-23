# 3. Full worked example: customer segmentation (banking-flavored)

# This mirrors the kind of thing you'd do for risk grouping or customer segmentation at Santander
# — say, clustering customers on spending behavior, account activity, and risk indicators.


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- synthetic "banking" dataset: 8 features, 500 customers ---
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'avg_balance': np.random.gamma(2, 2000, n),
    'monthly_txn_count': np.random.poisson(15, n),
    'avg_txn_amount': np.random.gamma(2, 50, n),
    'credit_utilization': np.random.beta(2, 5, n),
    'late_payments_12mo': np.random.poisson(0.5, n),
    'account_age_years': np.random.gamma(3, 2, n),
    'international_txn_pct': np.random.beta(1, 10, n),
    'overdraft_count_12mo': np.random.poisson(0.3, n),
})

# --- scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- PCA down to 2 components for both speed AND visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"2 components explain {pca.explained_variance_ratio_.sum():.1%} of variance")

# --- find a good k using silhouette score ---
best_k, best_score = None, -1
for k in range(2, 8):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}: silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score = k, score

# --- final clustering ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_pca)

# --- visualize ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='centroids')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title(f'Customer segments (k={best_k})')
plt.legend()
plt.colorbar(scatter, label='cluster')
plt.show()



# 4. Interpreting what each principal component means

# This is the part that trips people up — PCA components aren't your original features
# anymore, they're weighted combinations of them. To interpret a cluster, look at the loadings:

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=df.columns[:-1]  # exclude the cluster column
)
print(loadings)

# Which original features drive PC1 the most?
print(loadings['PC1'].abs().sort_values(ascending=False))


# If credit_utilization and late_payments_12mo have the largest weights on PC1, 
# you can say "PC1 roughly represents credit risk behavior" — and then describe clusters in business terms ("Cluster 2 = high balance, low risk" etc.) 
# rather than just "Cluster 2."


