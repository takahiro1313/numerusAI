# 数値AI（先行活動→来月の成約数予測）ハンズオンアプリ

「数字を入れたら数字が出る」を、fit（学習）/ predict（推論）を自分で書いて体験するStreamlitアプリ。
題材は **営業の先行活動 → 翌月の成約数を予測**（フォーキャスト・XGBoost）。「来月どれくらい取れそう？」に答える。

## ファイル

| ファイル | 役割 |
|---|---|
| `app_ensyu.py` | **演習用**（参加者に配布）。fit/predict が空欄（`___`） |
| `app_seikai.py` | **正解用**（講師リファレンス）。記入済みでそのまま動く |
| `ai_core.py` | 共通ロジック（データ・モデル・exec実行・描画） |
| `make_dummy.py` | ダミーCSV生成スクリプト |
| `data/activities.csv` | ダミー時系列（900行＝30担当×30ヶ月・SFDCイメージ） |
| `requirements.txt` | 依存（streamlit / xgboost / pandas / scikit-learn / numpy） |

## ローカル起動

```bash
pip install -r requirements.txt
python make_dummy.py            # data/activities.csv が無ければ生成
streamlit run app_ensyu.py     # 参加者用（空欄）
streamlit run app_seikai.py    # 講師用（正解・答え合わせ）
```

> Mac ローカルで xgboost が `libomp.dylib` 不足で動かない場合は `brew install libomp` を一度実行。
> Streamlit Community Cloud（Linux）では不要。

## 当日の触り方（演習用・スライド53 Step1-4）

1. **Step1 データを見る**（🔒）：担当×月の活動量と成約数（時系列）の形を確認
2. **Step2 学習する**（🟢）：`model.fit(___, ___)` の `___` に `X_train, y_train` を入れ → 🎓 学習ボタン
3. **Step3 推論する**（🟢）：営業担当を選び、`model.predict(___)` の `___` に `X_new` を入れ → 🔮 推論ボタン → その担当の翌月予測成約数
4. **Step4 パラメータを変える**（🟢）：`n_estimators` / `max_depth` を上げて学習し直す
   → 訓練スコア↑なのに検証スコア↓＝**過学習**を train/val 対比で体感（検証は時間分割＝未来の月）

> 正解は `app_seikai.py`（fit/predict 記入済み）。

## デプロイ（Streamlit Community Cloud）

- **Python は 3.11**（デプロイ時の Advanced settings で Python 3.11 を選択）。
- メイン実行ファイルは `app_ensyu.py`（受講生が**学習・推論を試す**アプリ。事前デプロイ必須）。
- **1人1URL（計5）** を別々にデプロイし、URLを配布。
- `data/activities.csv` をリポジトリに含めて配置（無い場合は `make_dummy.py` で生成してからコミット）。
- セッション前に**全URLをウォームアップ**。XGBoost を含むため初回ビルドに時間がかかる場合あり。
- 予備URLを1つ用意しておくと安心。

## 設計メモ

- 学習・推論はすべてサーバ側（数秒）。APIキー不要。
- fit/predict は限定名前空間で `exec`。`import`/I/O は渡さない。例外は画面表示。
- 過学習は train/val の R² 対比で可視化（検証は時間分割）。差が開くと警告を出す。
- 重要度で「架電（量）より提案・面談が翌月成約に効く」が見える。
- データは**ダミーのみ**。本番SFDC/TSR等は載せない。
- 詳細は `要件定義書.md` / `仕様設計書.md` を参照。
