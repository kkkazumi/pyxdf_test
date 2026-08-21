# graph drawing
## separate graph
`python3 test2.py`

select dir name

You can change the array of trigger just to make it easy to check.

## no-separate but with lines
`python3 test_line.py`

use the program to draw a graph with the line of triggers.

## Timeseries of HP values
`python3 HP_timeseries.py`
you will get a pic `000_HP_timeseries.*`
select `*.png` or `*.eps`

## statistics
to get statistic values of HP,
`python3 HP_statistics.py`
you will get `000_y_valuesHM.csv` and `000_y_valuesRB.csv`
HM is a data under the Human hint condition
RB is a data under the Robot hint condition

These two csv files are needed to generate pic when you execute `draw_HP_pic.py`

## HP_timeseries_analysis.py
複数ユーザの HP 時系列データ（NH / RB / HM）を読み込み、
B-spline（df を全ユーザで最適化）＋ 4 次多項式でフィッティングを行い、
以下を自動生成します：

- 全ユーザ共通の最適 df の推定
- HP の補間・平滑化
- Spline / Polynomial のフィッティング曲線
- 残差プロット
- フィッティング重み（特徴量）の保存

各ユーザフォルダに対して、
`*_HP_fit.png`, `*_HP_residuals.png`, `*_HP_basis_weights_all.csv`  
が出力されます。


## HP_timeseries_analysis_CV.py
こっちが本命
複数ユーザの HP 時系列データ（NH / RB / HM）を読み込み、
K-fold クロスバリデーション（CV）によって最適な B-spline の自由度 df を選択し、
選ばれた df を使って HP を B-spline + 4 次多項式でフィッティングします。

主な処理内容：
- 全ユーザのデータを使った df のクロスバリデーション（CV）
- HP の補間・平滑化（線形補間 + 移動平均）
- Spline / Polynomial のフィッティング
- フィッティング重み（特徴量）の保存
  - `*_HP_basis_weights_CVdfX.csv`
- フィッティンググラフの保存
  - `*_HP_fit_CVdfX.png`

## weight_PCA.py
このスクリプトは、CV により選択された df（例：df=15）で計算された
HP 特徴量（B-spline + Polynomial の重み）を読み込み、以下の解析を行います。
- 全ユーザ・全条件（NH / RB / HM）の重みベクトルを PCA によって 2 次元へ圧縮
- PCA の寄与率・loading の保存
- 条件ごとの PCA 座標（NH, RB, HM）を整理
- NH→RB、NH→HM の 変化ベクトル（方向ベクトル）を計算して CSV に保存
- 変化ベクトルを K-means（最適クラスタ数は silhouette で推定） によりクラスタリング
- NH→RB / NH→HM の 矢印プロット（クラスタ色付き） を生成

## weight_PCAmatome2.py
このスクリプトは、CV で選択された df（例：df=15）で計算された
HP 特徴量（B-spline + Polynomial の重み）を読み込み、
NH→RB / NH→HM の 変化ベクトル（RB−NH、HM−NH）を直接解析するためのものです。

主な処理内容：
- NH→RB / NH→HM の 変化ベクトル（21次元）を計算
- 変化ベクトルを 標準化（StandardScaler）
- RB と HM をまとめて K-means クラスタリング
  - クラスタ数は silhouette 係数で自動決定
- クラスタ割り当て・クラスタ中心・シルエット係数を CSV に保存
- 変化ベクトルに対して PCA（2次元）を実施
- RB と HM の PCA 座標を 同じ図に描画し、ユーザごとに線で接続
  - RB と HM の変化方向・変化量の比較が容易になる

## HP_clustering.py
このスクリプトは、CV で選択された df（例：df=15）で計算された
NH（通常状態）の HP 特徴量（B-spline 16 + Polynomial 5 = 21次元）を読み込み、
参加者間の構造を以下の手順で解析します。
一応これで、ノーヒントのパターンを見てるんだな。

主な処理内容：
- NH の特徴量（21次元）を読み込み
- 標準化（StandardScaler）
- K-means によるクラスタリング
  - クラスタ数は silhouette 係数で自動決定
- PCA（2次元）でクラスタを可視化
- クラスタごとの平均曲線＋標準偏差帯の描画
- クラスタ割り当て・クラスタ中心・PCA 寄与率・loading を CSV に保存

## HP_clustering_Q1skill.py
このスクリプトは、CVdf15 で算出された NH（通常状態）の HP 特徴量（21次元）を読み込み、
以下の解析を行います：
- NH 特徴量の 標準化
- K-means によるクラスタリング
  - クラスタ数は silhouette 係数で自動決定
- PCA（2次元）でクラスタを可視化
- Speech Skill（Q1）を色で表現
- クラスタ形状（△ / ▽）で表現
- クラスタ割り当て・クラスタ中心・PCA 寄与率・loading を CSV に保存
Q1のファイルはCSVで保存してる

## check_all_fitting_PDF.py
このスクリプトは、各参加者フォルダに保存された
NH / RB / HM の HP フィッティング画像（*_HP_fit.png） を読み込み、
全ユーザ分を 横向き（ランドスケープ）2×2 レイアウトの PDF にまとめるためのツールです。

