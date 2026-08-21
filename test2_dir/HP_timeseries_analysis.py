import numpy as np
from scipy.interpolate import interp1d
from patsy import dmatrix
import matplotlib.pyplot as plt

# --- AIC/BIC計算 ---
def compute_aic_bic(residuals, k):
    n = len(residuals)
    rss = np.sum(residuals**2)
    aic = n * np.log(rss/n) + 2 * k
    bic = n * np.log(rss/n) + k * np.log(n)
    return aic, bic

# --- 全ユーザの時系列を使って df を探索 ---
def find_best_df_all_users(dirnames):

    df_candidates = range(4, 16)  # ★ df=4〜15
    best_df = None
    best_rms = 1e18

    all_t = []
    all_hp = []

    for dirname in dirnames:
        for cond in ["NH", "RB", "HM"]:
            path = f"./{dirname}/{cond}_timeseries_raw.csv"
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            all_t.append(data[:,0])
            all_hp.append(data[:,1])

    for df in df_candidates:
        total_rss = 0
        total_n = 0

        for t, hp in zip(all_t, all_hp):

            t_new = np.arange(0, 60.0, 0.1)
            f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
            interpolated_hp = f_interp(t_new)

            smoothed_hp = np.convolve(interpolated_hp,
                                      np.ones(10)/10,
                                      mode='same')

            spline_basis = np.asarray(
                dmatrix(f"bs(t_new, df={df}, degree=3, include_intercept=False)",
                        {"t_new": t_new}, return_type='dataframe')
            )

            w_spline, _, _, _ = np.linalg.lstsq(spline_basis, smoothed_hp, rcond=None)
            fitted_spline = spline_basis @ w_spline

            fitted_on_t = np.interp(t, t_new, fitted_spline)
            residuals = hp - fitted_on_t

            total_rss += np.sum(residuals**2)
            total_n += len(residuals)

        rms = np.sqrt(total_rss / total_n)
        print(f"df={df}: RMS={rms:.4f}")

        if rms < best_rms:
            best_rms = rms
            best_df = df

    print(f"\n=== 全ユーザ共通の最適df = {best_df} (RMS={best_rms:.4f}) ===")
    return best_df

# --- 共通dfで特徴量とFittingを計算 ---
def compute_hp_weights(path, df):

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    t = data[:,0]
    hp = data[:,1]

    t_new = np.arange(0, 60.0, 0.1)
    f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
    interpolated_hp = f_interp(t_new)

    smoothed_hp = np.convolve(interpolated_hp,
                              np.ones(10)/10,
                              mode='same')

    spline_basis = np.asarray(
        dmatrix(f"bs(t_new, df={df}, degree=3, include_intercept=False)",
                {"t_new": t_new}, return_type='dataframe')
    )
    w_spline, _, _, _ = np.linalg.lstsq(spline_basis, smoothed_hp, rcond=None)
    fitted_spline = spline_basis @ w_spline
    fitted_spline_on_t = np.interp(t, t_new, fitted_spline)
    residual_spline = hp - fitted_spline_on_t

    poly_basis = np.vstack([
        np.ones_like(t_new),
        t_new,
        t_new**2,
        t_new**3,
        t_new**4
    ]).T
    w_poly, _, _, _ = np.linalg.lstsq(poly_basis, smoothed_hp, rcond=None)
    fitted_poly = poly_basis @ w_poly
    fitted_poly_on_t = np.interp(t, t_new, fitted_poly)
    residual_poly = hp - fitted_poly_on_t

    return {
        "t": t,
        "hp": hp,
        "t_new": t_new,
        "interpolated": interpolated_hp,
        "smoothed": smoothed_hp,
        "spline_basis": spline_basis,
        "w_spline": w_spline,
        "fitted_spline": fitted_spline,
        "residual_spline": residual_spline,
        "poly_basis": poly_basis,
        "w_poly": w_poly,
        "fitted_poly": fitted_poly,
        "residual_poly": residual_poly,
        "df": df
    }

# --- Fittingグラフ ---
def plot_fit(result, title, save_path):
    t = result["t"]
    hp = result["hp"]
    t_new = result["t_new"]
    interpolated = result["interpolated"]
    smoothed = result["smoothed"]
    spline_basis = result["spline_basis"]
    poly_basis = result["poly_basis"]
    w_spline = result["w_spline"]
    w_poly = result["w_poly"]
    df = result["df"]

    plt.figure(figsize=(10,5))
    plt.plot(t, hp, 'o', markersize=3, label="Original HP", alpha=0.6)
    plt.plot(t_new, interpolated, '-', label="Interpolated", alpha=0.5)
    plt.plot(t_new, smoothed, '-', label="Smoothed", linewidth=2)
    plt.plot(t_new, spline_basis @ w_spline, '-', label=f"Spline (df={df})", linewidth=2)
    plt.plot(t_new, poly_basis @ w_poly, '--', label="Polynomial 4th", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print("保存:", save_path)

# --- 残差グラフ ---
def plot_residual(result, title, save_path):
    t = result["t"]
    plt.figure(figsize=(10,4))
    plt.plot(t, result["residual_spline"], label="Residual (Spline)", alpha=0.7)
    plt.plot(t, result["residual_poly"], label="Residual (Polynomial)", alpha=0.7)
    plt.axhline(0, color="black")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print("保存:", save_path)

# --- メイン処理 ---
dirnames = ["001","003","013","015","018","030","031","034","035","040","043","044","045","046","047"]

# 1) 全ユーザで df を決める
best_df = find_best_df_all_users(dirnames)

# 2) 全ユーザの特徴量とグラフを作成
for dirname in dirnames:
    for cond in ["NH", "RB", "HM"]:
        path = f"./{dirname}/{cond}_timeseries_raw.csv"
        result = compute_hp_weights(path, best_df)

        # 重み保存
        weights_vector = np.hstack([result["w_spline"], result["w_poly"]])
        np.savetxt(f"./{dirname}/{cond}_HP_basis_weights_all.csv",
                   weights_vector.reshape(1, -1),
                   delimiter=",",
                   header="bspline1,bspline2,bspline3,bspline4,bspline5,bspline6,bspline7,poly1,poly2,poly3,poly4,poly5",
                   comments="")
        print(f"{dirname} {cond} 保存完了")

        # Fittingグラフ（★ NH / RB / HM の3枚）
        plot_fit(result,
                 f"{cond} HP Fit (df={best_df})",
                 f"./{dirname}/{cond}_HP_fit.png")

        # 残差グラフ
        plot_residual(result,
                      f"{cond} HP Residuals",
                      f"./{dirname}/{cond}_HP_residuals.png")
