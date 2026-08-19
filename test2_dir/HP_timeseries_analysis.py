import numpy as np
from scipy.interpolate import interp1d
from patsy import dmatrix
import matplotlib.pyplot as plt

# --- HP重みを計算する関数（NH/RB/HM共通） ---
def compute_hp_weights(timeseries_path):
    data = np.loadtxt(timeseries_path, delimiter=",", skiprows=1)
    t = data[:,0]
    hp = data[:,1]

    # 補間
    t_new = np.arange(0, 60.0, 0.1)
    f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
    interpolated_hp = f_interp(t_new)

    # 移動平均
    window_size = 10
    kernel = np.ones(window_size) / window_size
    smoothed_hp = np.convolve(interpolated_hp, kernel, mode='same')

    # B-spline
    spline_basis = np.asarray(
        dmatrix("bs(t_new, df=6, degree=3, include_intercept=False)",
                {"t_new": t_new}, return_type='dataframe')
    )
    w_spline, _, _, _ = np.linalg.lstsq(spline_basis, smoothed_hp, rcond=None)

    # Polynomial（4次）
    poly_basis = np.vstack([
        np.ones_like(t_new),
        t_new,
        t_new**2,
        t_new**3,
        t_new**4
    ]).T
    w_poly, _, _, _ = np.linalg.lstsq(poly_basis, smoothed_hp, rcond=None)

    return w_spline, w_poly, t, hp, t_new, interpolated_hp, smoothed_hp, spline_basis, poly_basis


# --- 1) dirname をユーザーに聞く ---
dirname = input("解析するディレクトリ名を入力してください： ")

# --- 2) NH / RB / HM の時系列データパス ---
nh_path = f"./{dirname}/NH_timeseries_raw.csv"
rb_path = f"./{dirname}/RB_timeseries_raw.csv"
hm_path = f"./{dirname}/HM_timeseries_raw.csv"

print("読み込み中:", nh_path)

# --- 3) NH / RB / HM の重みを計算 ---
nh_w_spline, nh_w_poly, t, hp, t_new, interpolated_hp, smoothed_hp, spline_basis, poly_basis = compute_hp_weights(nh_path)
rb_w_spline, rb_w_poly, *_ = compute_hp_weights(rb_path)
hm_w_spline, hm_w_poly, *_ = compute_hp_weights(hm_path)

# --- 4) 重みを保存（NH / RB / HM） ---
header = "bspline,poly"

def save_weights(name, w_spline, w_poly):
    # 横方向に結合（正しい）
    weights_vector = np.hstack([w_spline, w_poly])

    # 1行12列で保存
    np.savetxt(f"./{dirname}/{name}_HP_basis_weights_all.csv",
               weights_vector.reshape(1, -1),
               delimiter=",",
               header="bspline1,bspline2,bspline3,bspline4,bspline5,bspline6,bspline7,poly1,poly2,poly3,poly4,poly5",
               comments="")
    print(f"{name} の重みを保存しました:", f"./{dirname}/{name}_HP_basis_weights_all.csv")


save_weights("NH", nh_w_spline, nh_w_poly)
save_weights("RB", rb_w_spline, rb_w_poly)
save_weights("HM", hm_w_spline, hm_w_poly)

# --- 5) 差分（RB−NH, HM−NH）を保存 ---
rb_diff = np.hstack([rb_w_spline - nh_w_spline,
                     rb_w_poly - nh_w_poly])

hm_diff = np.hstack([hm_w_spline - nh_w_spline,
                     hm_w_poly - nh_w_poly])

np.savetxt(f"./{dirname}/RB_minus_NH_weights.csv",
           rb_diff.reshape(1, -1),
           delimiter=",",
           header="bspline1,bspline2,bspline3,bspline4,bspline5,bspline6,bspline7,poly1,poly2,poly3,poly4,poly5",
           comments="")

np.savetxt(f"./{dirname}/HM_minus_NH_weights.csv",
           hm_diff.reshape(1, -1),
           delimiter=",",
           header="bspline1,bspline2,bspline3,bspline4,bspline5,bspline6,bspline7,poly1,poly2,poly3,poly4,poly5",
           comments="")

print("差分（RB−NH, HM−NH）を保存しました")


# --- 6) グラフ描画（NHのみ） ---
plt.figure(figsize=(10,5))

plt.plot(t, hp, 'o', markersize=3, label="Original NH HP", alpha=0.6)
plt.plot(t_new, interpolated_hp, '-', label="Interpolated (0.1s)", alpha=0.5)
plt.plot(t_new, smoothed_hp, '-', label="Smoothed (1s MA)", linewidth=2)
plt.plot(t_new, spline_basis @ nh_w_spline, '-', label="Fitted (B-spline)", linewidth=2)
plt.plot(t_new, poly_basis @ nh_w_poly, '--', label="Fitted (Polynomial 4th)", linewidth=2)

plt.xlabel("Time [s]")
plt.ylabel("HP")
plt.title("NH HP: Original vs Interpolated vs Smoothed vs Spline vs Polynomial")
plt.legend()
plt.grid(True)

plt.savefig(f"./{dirname}/NH_HP_fit_spline_polynomial_comparison.png")
#plt.show()

print("比較グラフを保存しました:", f"./{dirname}/NH_HP_fit_spline_polynomial_comparison.png")
