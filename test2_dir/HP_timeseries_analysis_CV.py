import numpy as np
from scipy.interpolate import interp1d
from patsy import dmatrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# --- CV による df 選択 ---
def cv_select_df(dirnames, df_candidates=range(4, 16), n_splits=5):

    all_t = []
    all_hp = []

    for dirname in dirnames:
        for cond in ["NH", "RB", "HM"]:
            path = f"./{dirname}/{cond}_timeseries_raw.csv"
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            all_t.append(data[:,0])
            all_hp.append(data[:,1])

    all_t = np.array(all_t, dtype=object)
    all_hp = np.array(all_hp, dtype=object)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    df_rms_scores = {df: [] for df in df_candidates}

    for df in df_candidates:
        print(f"\n=== df={df} の CV 開始 ===")
        fold_rms = []

        for train_idx, test_idx in kf.split(all_hp):

            t_new = np.arange(0, 60.0, 0.1)
            spline_basis = np.asarray(
                dmatrix(f"bs(t_new, df={df}, degree=3, include_intercept=False)",
                        {"t_new": t_new}, return_type='dataframe')
            )

            smoothed_list = []
            for idx in train_idx:
                t = all_t[idx]
                hp = all_hp[idx]
                f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
                interpolated_hp = f_interp(t_new)
                smoothed_hp = np.convolve(interpolated_hp, np.ones(10)/10, mode='same')
                smoothed_list.append(smoothed_hp)

            smoothed_mean = np.mean(smoothed_list, axis=0)
            w_spline, _, _, _ = np.linalg.lstsq(spline_basis, smoothed_mean, rcond=None)

            fold_errors = []
            for idx in test_idx:
                t = all_t[idx]
                hp = all_hp[idx]

                f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
                interpolated_hp = f_interp(t_new)
                smoothed_hp = np.convolve(interpolated_hp, np.ones(10)/10, mode='same')

                fitted_spline = spline_basis @ w_spline
                fitted_on_t = np.interp(t, t_new, fitted_spline)
                residuals = hp - fitted_on_t

                rms = np.sqrt(np.mean(residuals**2))
                fold_errors.append(rms)

            fold_rms.append(np.mean(fold_errors))
            print(f" fold RMS = {np.mean(fold_errors):.4f}")

        df_rms_scores[df] = np.mean(fold_rms)
        print(f"df={df}: CV mean RMS={df_rms_scores[df]:.4f}")

    best_df = min(df_rms_scores, key=df_rms_scores.get)
    print(f"\n=== CV による最適 df = {best_df} ===")

    return best_df


# --- 重み計算 ---
def compute_hp_weights(path, df):

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    t = data[:,0]
    hp = data[:,1]

    t_new = np.arange(0, 60.0, 0.1)
    f_interp = interp1d(t, hp, kind='linear', fill_value="extrapolate")
    interpolated_hp = f_interp(t_new)

    smoothed_hp = np.convolve(interpolated_hp, np.ones(10)/10, mode='same')

    spline_basis = np.asarray(
        dmatrix(f"bs(t_new, df={df}, degree=3, include_intercept=False)",
                {"t_new": t_new}, return_type='dataframe')
    )
    w_spline, _, _, _ = np.linalg.lstsq(spline_basis, smoothed_hp, rcond=None)

    poly_basis = np.vstack([
        np.ones_like(t_new),
        t_new,
        t_new**2,
        t_new**3,
        t_new**4
    ]).T
    w_poly, _, _, _ = np.linalg.lstsq(poly_basis, smoothed_hp, rcond=None)

    return {
        "t": t,
        "hp": hp,
        "t_new": t_new,
        "interpolated": interpolated_hp,
        "smoothed": smoothed_hp,
        "spline_basis": spline_basis,
        "w_spline": w_spline,
        "poly_basis": poly_basis,
        "w_poly": w_poly,
        "df": df
    }


# --- Fitting グラフ ---
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
    plt.plot(t, result["interpolated"] - result["smoothed"], label="Residual (Spline)", alpha=0.7)
    plt.axhline(0, color="black")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print("保存:", save_path)


# --- メイン処理 ---
dirnames = ["001","003","013","015","018","030","031","034","035",
            "040","043","044","045","046","047"]

best_df = cv_select_df(dirnames)

for dirname in dirnames:
    for cond in ["NH", "RB", "HM"]:
        path = f"./{dirname}/{cond}_timeseries_raw.csv"
        result = compute_hp_weights(path, best_df)

        w_spline = result["w_spline"]   # df 次元
        w_poly   = result["w_poly"]     # 5 次元

        weights_vector = np.hstack([w_spline, w_poly])

        # --- df に応じてヘッダを自動生成 ---
        header_spline = [f"bspline{i+1}" for i in range(len(w_spline))]
        header_poly   = [f"poly{i+1}"    for i in range(len(w_poly))]
        header_all    = header_spline + header_poly
        header_str    = ",".join(header_all)

        save_name = f"./{dirname}/{cond}_HP_basis_weights_CVdf{best_df}.csv"

        np.savetxt(
            save_name,
            weights_vector.reshape(1, -1),
            delimiter=",",
            header=header_str,
            comments=""
        )

        print(f"{dirname} {cond} 保存完了 → {save_name}")

        plot_fit(result,
                 f"{cond} HP Fit (df={best_df})",
                 f"./{dirname}/{cond}_HP_fit_CVdf{best_df}.png")
