"""
Single Linkage vs Complete Linkage — Dendrogram Demo
------------------------------------------------------
Dataset: sklearn.datasets.make_blobs (a small, synthetic dataset that's
"handy" -- no download needed, ships with scikit-learn).

We generate 3 well-separated blobs of points, then add one "bridge"
point between two of the blobs. That bridge point is what makes the
difference between single and complete linkage obvious:
  - Single linkage will "chain" through the bridge point and merge
    two blobs together too early.
  - Complete linkage resists this because it looks at the worst-case
    (farthest) distance, so it keeps the blobs separate longer.

Requires: scikit-learn, scipy, matplotlib
    pip install scikit-learn scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------------------------------
# 1. Create a small synthetic dataset (3 clear blobs)
# ---------------------------------------------------------
X, y_true = make_blobs(
    n_samples=30,
    centers=3,
    cluster_std=0.60,
    random_state=42,
)

# Add a single "bridge" point roughly between two of the blobs.
# This is what makes single-linkage "chaining" visible.
bridge_point = np.array([[np.mean(X[:, 0]) - 1.5, np.mean(X[:, 1])]])
X = np.vstack([X, bridge_point])

# Simple labels so we can see which original point is which in the dendrogram
labels = [f"P{i}" for i in range(len(X) - 1)] + ["BRIDGE"]

# ---------------------------------------------------------
# 2. Scale the data (always a good idea before clustering,
#    since linkage relies on distances)
# ---------------------------------------------------------
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------------
# 3. Compute linkage matrices for both methods
# ---------------------------------------------------------
mergings_single = linkage(X_scaled, method="single")
mergings_complete = linkage(X_scaled, method="complete")

# ---------------------------------------------------------
# 4. Plot both dendrograms side by side
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dendrogram(
    mergings_single,
    labels=labels,
    ax=axes[0],
    leaf_rotation=90,
    leaf_font_size=8,
)
axes[0].set_title("Single Linkage\n(chains through the bridge point)")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Distance")

dendrogram(
    mergings_complete,
    labels=labels,
    ax=axes[1],
    leaf_rotation=90,
    leaf_font_size=8,
)
axes[1].set_title("Complete Linkage\n(keeps clusters compact/separate longer)")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Distance")

plt.tight_layout()
plt.savefig("linkage_comparison.png", dpi=150)
print("Saved plot to linkage_comparison.png")
plt.show()