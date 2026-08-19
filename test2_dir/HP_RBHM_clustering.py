import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

valid_ids = [
    "001","003","013","015","018",
    "030","031","034","035","040",
    "043","044","045","046","047"
]

def load_weights(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data.flatten()  # 12次元

# --- 差分ベクトル作成 ---
participants = []
rb_diff_vectors = []
hm_diff_vectors = []

for pid in valid_ids:
    nh = load_weights(f"./{pid}/NH_HP_basis_weights_all.csv")
    rb = load_weights(f"./{pid}/RB_HP_basis_weights_all.csv")
    hm = load_weights(f"./{pid}/HM_HP_basis_weights_all.csv")

    rb_diff = rb - nh
    hm_diff = hm - nh

    rb_diff_vectors.append(rb_diff)
    hm_diff_vectors.append(hm_diff)
    participants.append(pid)

rb_diff_vectors = np.array(rb_diff_vectors)
hm_diff_vectors = np.array(hm_diff_vectors)

# --- Combined 24次元ベクトル ---
combined_vectors = np.hstack([rb_diff_vectors, hm_diff_vectors])

print("読み込んだ参加者数:", len(participants))
print("特徴次元:", combined_vectors.shape[1])

# --- 標準化 ---
scaler = StandardScaler()
X = scaler.fit_transform(combined_vectors)

# --- シルエット係数で最適クラスタ数を決定 ---
sil_scores = {}

for k in range(2, 7):  # 2〜6クラスタを試す
    kmeans_tmp = KMeans(n_clusters=k, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(X)
    score = silhouette_score(X, labels_tmp)
    sil_scores[k] = score
    print(f"K={k}, Silhouette Score={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n最適クラスタ数（Silhouette基準）: K={best_k}")

# --- 最適クラスタ数でクラスタリング ---
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)

# --- コンソール出力 ---
print("\n=== RB-NH + HM-NH Combined Clustering ===")
for pid, lab in zip(participants, labels):
    print(f"{pid} → クラスタ {lab}")

# --- グラフ（24次元） ---
plt.figure(figsize=(14,6))
dims = np.arange(24)

for c in range(best_k):
    cluster_vecs = combined_vectors[labels == c]
    mean_vec = cluster_vecs.mean(axis=0)
    std_vec = cluster_vecs.std(axis=0)

    plt.plot(dims, mean_vec, linewidth=3, label=f"Cluster {c}")

    for vec in cluster_vecs:
        plt.plot(dims, vec, alpha=0.15)

    plt.fill_between(dims,
                     mean_vec - std_vec,
                     mean_vec + std_vec,
                     alpha=0.1)

# --- 24次元すべてに縦線 ---
for x in range(24):
    plt.axvline(x, color="lightgray", linewidth=0.5, alpha=0.6)

# RBとHMの境界
plt.axvline(12, color="black", linewidth=1.5)

plt.title("Combined RB-NH + HM-NH Clustering (Mean + Individual + Std Band)")
plt.xlabel("Dimension (RB diff 1-12, HM diff 13-24)")
plt.ylabel("Difference Value")
plt.legend()
plt.grid(True)

plt.savefig("./Combined_RB_HM_cluster_band.png")
plt.show()

print("Combined クラスタ帯グラフを保存しました: ./Combined_RB_HM_cluster_band.png")

# --- 2次元プロット用の値 ---
rb_mean = rb_diff_vectors.mean(axis=1)
hm_mean = hm_diff_vectors.mean(axis=1)

# --- 2次元散布図 ---
plt.figure(figsize=(8,6))

colors = ["blue", "purple", "green", "orange", "red", "cyan"]

for c in range(best_k):
    idx = (labels == c)
    plt.scatter(
        hm_mean[idx],   # 横軸：HM−NH
        rb_mean[idx],   # 縦軸：RB−NH
        c=colors[c],
        label=f"Cluster {c}",
        alpha=0.7
    )

    # --- ID描画 ---
    for i in np.where(idx)[0]:
        plt.text(
            hm_mean[i],
            rb_mean[i],
            participants[i],
            fontsize=9,
            ha='center',
            va='center'
        )

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("HM - NH")
plt.ylabel("RB - NH")
plt.title("2D Plot: HM-NH vs RB-NH")
plt.legend()
plt.grid(True)

plt.savefig("./HM_RB_2D_plot.png")
plt.show()

print("2次元プロットを保存しました: ./HM_RB_2D_plot.png")

import matplotlib.pyplot as plt
import numpy as np

# --- RB第3基底（index=2）と HM第3基底（index=14）を抽出 ---
rb_bspline3 = rb_diff_vectors[:, 2]     # RB−NH の B-spline 第3基底
hm_bspline3 = hm_diff_vectors[:, 2]     # HM−NH の B-spline 第3基底

plt.figure(figsize=(8,6))
plt.boxplot([rb_bspline3, hm_bspline3])
plt.xticks([1, 2], ["RB-NH (Bspline3)", "HM-NH (Bspline3)"])
plt.title("Boxplot: RB vs HM (B-spline 3rd basis)")
plt.ylabel("Difference Value")
plt.grid(True)
plt.savefig("./boxplot_rb_hm_bspline3.png")
plt.show()


# --- ヒストグラム（Histogram） ---
plt.figure(figsize=(8,6))
plt.hist(rb_bspline3, bins=10, alpha=0.6, label="RB-NH (Bspline3)")
plt.hist(hm_bspline3, bins=10, alpha=0.6, label="HM-NH (Bspline3)")
plt.title("Histogram: RB vs HM (B-spline 3rd basis)")
plt.xlabel("Difference Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.savefig("./hist_rb_hm_bspline3.png")
plt.show()
