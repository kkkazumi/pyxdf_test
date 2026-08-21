import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import csv

# --- 対象ユーザ ---
dirnames = ["001","003","013","015","018","030","031","034","035",
            "040","043","044","045","046","047"]

# --- 重み読み込み（★ CVdf15 を読むように変更） ---
weights = []
labels = []
users = []

for dirname in dirnames:
    for cond in ["NH", "RB", "HM"]:
        path = f"./{dirname}/{cond}_HP_basis_weights_CVdf15.csv"
        w = np.loadtxt(path, delimiter=",", skiprows=1)
        weights.append(w)
        labels.append(cond)
        users.append(dirname)

weights = np.array(weights)

# --- センタリング ---
mean_vec = np.mean(weights, axis=0)
weights_centered = weights - mean_vec

# --- PCA 2次元 ---
pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(weights_centered)

# ★ 寄与率を出力
explained = pca2.explained_variance_ratio_
print("PC1 寄与率:", explained[0])
print("PC2 寄与率:", explained[1])
print("累積寄与率:", explained.sum())

# ★ 寄与率を CSV に保存
with open("PCA_explained_variance_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1_explained_variance", explained[0]])
    writer.writerow(["PC2_explained_variance", explained[1]])
    writer.writerow(["cumulative", explained.sum()])

print("寄与率を PCA_explained_variance_CVdf15.csv に保存しました")

# --- loading を CSV に保存 ---
pc1_loading = pca2.components_[0]
pc2_loading = pca2.components_[1]

with open("PCA_loading_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1_loading"])
    writer.writerow(pc1_loading)
    writer.writerow(["PC2_loading"])
    writer.writerow(pc2_loading)

print("PCA loading を PCA_loading_CVdf15.csv に保存しました")

# --- 条件ごとの座標 ---
cond_index = {"NH": {}, "RB": {}, "HM": {}}
for x, cond, user in zip(X2, labels, users):
    cond_index[cond][user] = x

# --- 変化ベクトル抽出 ---
rows = []  # CSV 出力用

rb_vec = []
hm_vec = []
rb_users = []
hm_users = []

for dirname in dirnames:
    if dirname in cond_index["NH"]:
        nh = cond_index["NH"][dirname]

        dPC1_NHRB = ""
        dPC2_NHRB = ""
        dPC1_NHHM = ""
        dPC2_NHHM = ""

        if dirname in cond_index["RB"]:
            rb = cond_index["RB"][dirname]
            vec_rb = rb - nh
            rb_vec.append(vec_rb)
            rb_users.append(dirname)
            dPC1_NHRB = vec_rb[0]
            dPC2_NHRB = vec_rb[1]

        if dirname in cond_index["HM"]:
            hm = cond_index["HM"][dirname]
            vec_hm = hm - nh
            hm_vec.append(vec_hm)
            hm_users.append(dirname)
            dPC1_NHHM = vec_hm[0]
            dPC2_NHHM = vec_hm[1]

        rows.append([dirname, dPC1_NHRB, dPC2_NHRB, dPC1_NHHM, dPC2_NHHM])

# --- 変化ベクトル CSV 出力 ---
with open("PCA_change_vectors_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "dPC1_NHRB", "dPC2_NHRB", "dPC1_NHHM", "dPC2_NHHM"])
    writer.writerows(rows)

print("変化ベクトルを PCA_change_vectors_CVdf15.csv に保存しました")

# --- クラスタリング ---
def best_k(vectors):
    best_score = -1
    best_k = 2
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(vectors)
        score = silhouette_score(vectors, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

rb_vec = np.array(rb_vec)
hm_vec = np.array(hm_vec)

rb_k = best_k(rb_vec)
hm_k = best_k(hm_vec)

rb_kmeans = KMeans(n_clusters=rb_k, random_state=42).fit(rb_vec)
hm_kmeans = KMeans(n_clusters=hm_k, random_state=42).fit(hm_vec)

rb_labels = rb_kmeans.labels_
hm_labels = hm_kmeans.labels_

palette = ["red", "green", "blue", "purple", "orange"]

# --- NH→RB の図 ---
plt.figure(figsize=(12, 10))

for user in rb_users:
    nh = cond_index["NH"][user]
    rb = cond_index["RB"][user]
    plt.scatter(nh[0], nh[1], c="blue", s=80)
    plt.scatter(rb[0], rb[1], c="red", s=80)
    plt.text(nh[0], nh[1], user, fontsize=9)
    plt.text(rb[0], rb[1], user, fontsize=9)

for dirname, vec, lab in zip(rb_users, rb_vec, rb_labels):
    nh = cond_index["NH"][dirname]
    rb = cond_index["RB"][dirname]
    plt.arrow(nh[0], nh[1],
              rb[0] - nh[0], rb[1] - nh[1],
              color=palette[lab],
              alpha=0.7,
              width=0.003,
              head_width=0.03,
              head_length=0.04,
              length_includes_head=True)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NH→RB 矢印（方向ベクトルクラスタリング）")
plt.grid(True)
plt.savefig("PCA_arrow_RB_clustered_CVdf15.png")
plt.close()

# --- NH→HM の図 ---
plt.figure(figsize=(12, 10))

for user in hm_users:
    nh = cond_index["NH"][user]
    hm = cond_index["HM"][user]
    plt.scatter(nh[0], nh[1], c="blue", s=80)
    plt.scatter(hm[0], hm[1], c="green", s=80)
    plt.text(nh[0], nh[1], user, fontsize=9)
    plt.text(hm[0], hm[1], user, fontsize=9)

for dirname, vec, lab in zip(hm_users, hm_vec, hm_labels):
    nh = cond_index["NH"][dirname]
    hm = cond_index["HM"][dirname]
    plt.arrow(nh[0], nh[1],
              hm[0] - nh[0], hm[1] - nh[1],
              color=palette[lab],
              alpha=0.7,
              width=0.003,
              head_width=0.03,
              head_length=0.04,
              length_includes_head=True)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NH→HM 矢印（方向ベクトルクラスタリング）")
plt.grid(True)
plt.savefig("PCA_arrow_HM_clustered_CVdf15.png")
plt.close()

print("RB と HM の矢印クラスタリング図を保存しました。")
