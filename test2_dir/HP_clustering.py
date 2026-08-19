import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

valid_ids = [
    "001","003","013","015","018",
    "030","031","034","035","040",
    "043","044","045","046","047"
]

participants = []
vectors = []

for pid in valid_ids:
    path = f"./{pid}/NH_HP_basis_weights_all.csv"

    try:
        df = pd.read_csv(path)
    except:
        print(f"⚠ {pid} の NH_HP_basis_weights_all.csv が見つかりません")
        continue

    bspline = df["bspline"].values
    poly = df["poly"].values

    # NaN除去
    bspline = bspline[~np.isnan(bspline)]
    poly = poly[~np.isnan(poly)]

    if len(bspline) != 7 or len(poly) != 5:
        print(f"⚠ {pid} の重みの長さが異常です: bspline={len(bspline)}, poly={len(poly)}")
        continue

    vec = np.concatenate([bspline, poly])
    vectors.append(vec)
    participants.append(pid)

vectors = np.array(vectors)

print("読み込んだ参加者数:", vectors.shape[0])
print("特徴次元:", vectors.shape[1])

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(vectors)

# k-means
K = 3
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(X)

# --- コンソール出力：どのIDがどのクラスタか ---
print("\n=== クラスタリング結果 ===")
for pid, lab in zip(participants, labels):
    print(f"参加者 {pid} → クラスタ {lab}")

# 結果保存
result_df = pd.DataFrame({
    "participant": participants,
    "cluster": labels
})
result_df.to_csv("./NH_clustering_result.csv", index=False)
print("\nクラスタリング結果を保存しました: ./NH_clustering_result.csv")

# --- クラスタごとの平均＋帯プロット ---
plt.figure(figsize=(12,6))

dims = np.arange(12)

for c in range(K):
    cluster_vecs = vectors[labels == c]

    # 平均線
    mean_vec = cluster_vecs.mean(axis=0)
    plt.plot(dims, mean_vec, linewidth=3, label=f"Cluster {c}")

    # 個々の線（薄くプロット）
    for vec in cluster_vecs:
        plt.plot(dims, vec, color=plt.gca().lines[-1].get_color(), alpha=0.2)

    # 帯（標準偏差）
    std_vec = cluster_vecs.std(axis=0)
    plt.fill_between(dims,
                     mean_vec - std_vec,
                     mean_vec + std_vec,
                     color=plt.gca().lines[-1].get_color(),
                     alpha=0.1)

plt.title("NH Clusters: Mean + Individual Curves + Std Band")
plt.xlabel("Dimension (Bspline 1-7, Poly 1-5)")
plt.ylabel("Weight Value")
plt.legend()
plt.grid(True)

plt.savefig("./NH_cluster_mean_band.png")
plt.show()

print("クラスタ帯グラフを保存しました: ./NH_cluster_mean_band.png")
