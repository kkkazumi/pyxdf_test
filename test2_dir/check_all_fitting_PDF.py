import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# --- 解析対象のユーザID ---
dirnames = ["001","003","013","015","018","030","031","034","035","040","043","044","045","046","047"]

# --- PDF 出力 ---
output_pdf = "AllUsers_HP_Fitting_2x2.pdf"

with PdfPages(output_pdf) as pdf:

    for dirname in dirnames:
        fig = plt.figure(figsize=(16, 11))  # ★ 横向き（ランドスケープ）

        # NH / RB / HM の画像パス
        nh_img = f"./{dirname}/NH_HP_fit.png"
        rb_img = f"./{dirname}/RB_HP_fit.png"
        hm_img = f"./{dirname}/HM_HP_fit.png"

        img_paths = [nh_img, rb_img, hm_img]

        # --- 2×2 の枠に配置 ---
        for i, img_path in enumerate(img_paths):
            ax = fig.add_subplot(2, 2, i+1)
            if os.path.exists(img_path):
                ax.imshow(plt.imread(img_path))
                ax.axis("off")
                ax.set_title(f"{dirname} - {['NH','RB','HM'][i]}", fontsize=14)
            else:
                ax.text(0.5, 0.5, f"{img_path} not found",
                        ha="center", va="center", fontsize=12)
                ax.axis("off")

        # --- 4枠目はユーザ情報を表示 ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5,
                 f"User: {dirname}\nNH / RB / HM HP Fitting\n(df = 共通値)",
                 ha="center", va="center", fontsize=16)
        ax4.axis("off")

        # PDFに保存
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF を出力しました: {output_pdf}")
