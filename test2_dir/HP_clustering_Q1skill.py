import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

dirnames = ["001","003","013","015","018","030","031","034","035",
            "040","043","044","045","046","047"]

# --- Speech Skill（Q1） ---
speech_skill_dict = {
    "001": 0,
    "003": 0.5,
    "013": np.nan,
    "015": 0,
    "018": 0.5,
    "030": -1,
    "031": -0.5,
    "034": -0.5,
    "035": -1,
    "040": -1,
    "043": -1,
    "044": -1,
    "045": -1,
    "046": -1,
    "047": -1
}

participants = []
vectors = []

# --- 重み読み込み ---
for dirname in dirnames:
    path = f"./{dirname}/NH_HP_basis_weights_CVdf15.csv"

    try:
        w = np.loadtxt(path, delimiter=",", skiprows=1)
    except:
        print(f"⚠ {dirname} の NH 重みが読み込めません")
        continue

    w = w[~np.isnan(w)]
    w = w[:21]  # bspline16 + poly5

    vectors.append(w)
    participants.append(dirname)

vectors = np.array(vectors)

print("読み込んだ参加者数:", vectors.shape[0])
print("特徴次元:", vectors.shape[1])

# --- 標準化 ---
scaler = StandardScaler()
X = scaler.fit_transform(vectors)

# --- シルエット係数で最適クラスタ数 ---
sil_scores = {}
for k in range(2, 7):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(X)
    score = silhouette_score(X, labels_tmp)
    sil_scores[k] = score
    print(f"[NH] K={k}, Silhouette Score={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n最適NHクラスタ数: K={best_k}")

# --- KMeansクラスタリング ---
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)

# --- PCA 2次元 ---
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

# ============================================================
# ★★★ Speech Skill（色）＋ Cluster（形）PCAプロット ★★★
# ============================================================

plt.figure(figsize=(10,8))

# 色：青→赤
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=-1, vmax=1)

xs = X2[:, 0]
ys = X2[:, 1]

# NaN は 0（中間色）に置き換え
skills = np.array([
    speech_skill_dict[pid] if not np.isnan(speech_skill_dict[pid]) else 0
    for pid in participants
])

clusters = labels

# --- Cluster 0（△） ---
idx0 = np.where(clusters == 0)[0]
sc0 = plt.scatter(xs[idx0], ys[idx0],
                  c=skills[idx0],
                  cmap=cmap, norm=norm,
                  marker="^", s=180, edgecolor="black")

# --- Cluster 1（▽） ---
idx1 = np.where(clusters == 1)[0]
sc1 = plt.scatter(xs[idx1], ys[idx1],
                  c=skills[idx1],
                  cmap=cmap, norm=norm,
                  marker="v", s=180, edgecolor="black")

# --- ラベル ---
for i, pid in enumerate(participants):
    plt.text(xs[i] + 0.03, ys[i] + 0.03, pid, fontsize=10)

plt.title("NH PCA + Cluster Shape (△/▽) + Speech Skill Color")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# --- colorbar（Cluster 0 の scatter を参照） ---
plt.colorbar(sc0, label="Speech Skill (Q1)")

plt.savefig("./NH_cluster_PCA2D_SpeechSkill.png")
plt.close()

print("改造版 PCA 図を保存しました: NH_cluster_PCA2D_SpeechSkill.png")

# ============================================================
# ここから下は元コードのまま（クラスタ保存など）
# ============================================================

# 結果保存
df = pd.DataFrame({"participant": participants, "cluster": labels})
df.to_csv("./NH_cluster_result_CVdf15.csv", index=False)

# --- NHクラスタ割り当て一覧を CSV に保存 ---
with open("NH_cluster_assignment_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["participant", "cluster"])
    for pid, lab in zip(participants, labels):
        writer.writerow([pid, lab])

print("\n=== NHクラスタ割り当て ===")
for pid, lab in zip(participants, labels):
    print(f"{pid}: Cluster {lab}")

# --- KMeansクラスタ中心（Loading）を保存 ---
cluster_centers = kmeans.cluster_centers_

with open("NH_cluster_centers_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"dim{i+1}" for i in range(cluster_centers.shape[1])]
    writer.writerow(["cluster"] + header)
    for idx, center in enumerate(cluster_centers):
        writer.writerow([idx] + list(center))

# --- PCA 寄与率の保存 ---
explained = pca.explained_variance_ratio_
cum_explained = explained.sum()

with open("NH_PCA_explained_variance_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1_explained_variance", explained[0]])
    writer.writerow(["PC2_explained_variance", explained[1]])
    writer.writerow(["cumulative_explained_variance", cum_explained])

pc1_loading = pca.components_[0]
pc2_loading = pca.components_[1]

with open("NH_PCA_loading_CVdf15.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PC1_loading"] + list(pc1_loading))
    writer.writerow(["PC2_loading"] + list(pc2_loading))
