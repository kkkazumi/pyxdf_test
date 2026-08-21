import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

# --- 対象ユーザ ---
dirnames = ["001","003","013","015","018","030","031","034","035",
            "040","043","044","045","046","047"]

# --- NH / RB / HM の重み読み込み ---
nh_weights = {}
rb_weights = {}
hm_weights = {}

for dirname in dirnames:
    nh = np.loadtxt(f"./{dirname}/NH_HP_basis_weights_CVdf15.csv",
                    delimiter=",", skiprows=1)
    rb = np.loadtxt(f"./{dirname}/RB_HP_basis_weights_CVdf15.csv",
                    delimiter=",", skiprows=1)
    hm = np.loadtxt(f"./{dirname}/HM_HP_basis_weights_CVdf15.csv",
                    delimiter=",", skiprows=1)

    nh_weights[dirname] = nh[:21]
    rb_weights[dirname] = rb[:21]
    hm_weights[dirname] = hm[:21]

# --- NH→RB / NH→HM の変化ベクトル ---
change_vectors = []
change_labels = []  # RB or HM
change_users = []

for dirname in dirnames:
    vec_rb = rb_weights[dirname] - nh_weights[dirname]
    vec_hm = hm_weights[dirname] - nh_weights[dirname]

    change_vectors.append(vec_rb)
    change_labels.append("RB")
    change_users.append(dirname)

    change_vectors.append(vec_hm)
    change_labels.append("HM")
    change_users.append(dirname)

change_vectors = np.array(change_vectors)

# --- 標準化 ---
scaler = StandardScaler()
X = scaler.fit_transform(change_vectors)

# --- クラスタ数決定（シルエット係数） ---
best_score = -1
best_k = 2
for k in range(2, 7):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42).fit(X)
    score = silhouette_score(X, kmeans_tmp.labels_)
    if score > best_score:
        best_score = score
        best_k = k

print(f"最適クラスタ数: {best_k}")
print(f"シルエット係数: {best_score}")

# --- クラスタリング ---
kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# --- クラスタ結果保存 ---
with open("change_cluster_assignment_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Condition", "Cluster"])
    for user, cond, lab in zip(change_users, change_labels, cluster_labels):
        writer.writerow([user, cond, lab])

with open("change_cluster_centers_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"dim{i+1}" for i in range(cluster_centers.shape[1])]
    writer.writerow(["cluster"] + header)
    for idx, center in enumerate(cluster_centers):
        writer.writerow([idx] + list(center))

with open("change_silhouette_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["best_k", best_k])
    writer.writerow(["silhouette_score", best_score])

print("クラスタリング結果を保存しました")

# --- PCA（変化ベクトルをまとめて） ---
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

explained = pca.explained_variance_ratio_

with open("change_PCA_explained_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1", explained[0]])
    writer.writerow(["PC2", explained[1]])
    writer.writerow(["cumulative", explained.sum()])

pc1_loading = pca.components_[0]
pc2_loading = pca.components_[1]

with open("change_PCA_loading_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1_loading"] + list(pc1_loading))
    writer.writerow(["PC2_loading"] + list(pc2_loading))

print("PCA 寄与率と loading を保存しました")

# --- Change PCA（RB と HM を同じ図に描く + 線でつなぐ） ---
plt.figure(figsize=(12, 10))
palette = ["red", "green", "blue", "purple", "orange"]

# RB と HM の座標を辞書にまとめる
rb_points = {}
hm_points = {}

for user, cond, lab, vec2 in zip(change_users, change_labels, cluster_labels, X2):
    color = palette[lab]
    if cond == "RB":
        rb_points[user] = vec2
        plt.scatter(vec2[0], vec2[1], c=color, marker="o", s=150)
        plt.text(vec2[0] + 0.03, vec2[1] + 0.03, f"{user}-RB", fontsize=10)
    else:
        hm_points[user] = vec2
        plt.scatter(vec2[0], vec2[1], c=color, marker="s", s=150)
        plt.text(vec2[0] + 0.03, vec2[1] + 0.03, f"{user}-HM", fontsize=10)

# --- RB と HM を線でつなぐ ---
for user in dirnames:
    if user in rb_points and user in hm_points:
        x_rb, y_rb = rb_points[user]
        x_hm, y_hm = hm_points[user]
        plt.plot([x_rb, x_hm], [y_rb, y_hm], c="gray", alpha=0.6)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NH→RB / NH→HM 変化ベクトル PCA（RB-HM を線で接続）")
plt.grid(True)
plt.savefig("change_PCA_RB_HM_connected_CVdf15.png")
plt.close()

print("RB-HM を線でつないだ PCA 図を保存しました")
