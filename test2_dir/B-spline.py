import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix

# 時間軸（あなたの解析と同じ）
t_new = np.arange(0, 60.0, 0.1)

# Bスプライン基底関数（df=6, degree=3）
spline_basis = np.asarray(
    dmatrix("bs(t_new, df=6, degree=3, include_intercept=False)",
            {"t_new": t_new}, return_type='dataframe')
)

# 基底関数の数
num_basis = spline_basis.shape[1]

plt.figure(figsize=(12, 6))

# 全基底関数をプロット
for i in range(num_basis):
    plt.plot(t_new, spline_basis[:, i], label=f"Basis {i+1}")

plt.title("B-spline Basis Functions (df=6, degree=3)")
plt.xlabel("Time [s]")
plt.ylabel("Basis value")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("B_spline_basis_functions.png")
plt.show()

print("Bスプライン基底関数を図示しました： B_spline_basis_functions.png")
