"""数値AI（先行活動→翌月の成約数）ハンズオン 共通ロジック。

app_ensyu.py（fit/predict が空欄）と app_seikai.py（記入済み）が、
スニペット定数だけ差し替えて render_app() を呼ぶ。

題材（要件 §4）:
- 営業担当 × 月 の時系列パネル。
- 先行活動（問い合わせ/架電/面談/案件提案）の「当月・1ヶ月前・2ヶ月前」を特徴量に、
  **翌月の成約数**を予測する（フォーキャスト）。結果系の契約数は使わない（自明回避）。

設計の要点（仕様設計書 §1.3 / §6）:
- 🟢 fit/predict の編集セルとパラメータのスライダーだけを参加者に触らせる。
- 🔒 データ読込・ラグ特徴量づくり・時間分割・スコア計算・グラフ描画は固定領域。
- 編集コードは限定名前空間で exec し、例外は画面表示（落とさない）。
- 過学習は train/val スコア対比で可視化（検証は「過去で学習→未来で検証」の時間分割）。
"""

import os
import streamlit as st
import pandas as pd

# 先行活動（結果系=契約数は含めない）
ACT_COLS = ["問い合わせ数", "架電数", "面談数", "案件提案数"]
LAGS = [("当月", 0), ("1ヶ月前", 1), ("2ヶ月前", 2)]
# ラグ特徴量の列名（12列）
FEATURE_COLS = [f"{a}_{label}" for a in ACT_COLS for (label, _) in LAGS]
TARGET_COL = "成約数"
TARGET_NEXT = "翌月成約数"   # 予測対象（翌月にズラした成約数）
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "activities.csv")

# 時間分割：終盤 VAL_MONTHS ヶ月を検証（未来）に回す
VAL_MONTHS = 6

# --- 🔒 安全な実行のための最小ビルトイン（仕様設計書 §6）-----------------
SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "len": len, "range": range, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "print": print, "round": round,
}


# --- 🔒 データ・特徴量・モデル ---------------------------------------------
def load_data():
    return pd.read_csv(DATA_PATH)


def build_features(df):
    """生の時系列パネル → ラグ特徴量＋翌月成約。

    各 営業担当 ごとに、活動の当月/1ヶ月前/2ヶ月前を作り、
    target = 成約数の「翌月」値（shift(-1)）にする。
    先頭2ヶ月（ラグ無し）と各担当の最終月（翌月成約が無い）は落とす。
    """
    df = df.sort_values(["営業担当", "月"]).copy()
    g = df.groupby("営業担当", group_keys=False)
    for act in ACT_COLS:
        for (label, lag) in LAGS:
            df[f"{act}_{label}"] = g[act].shift(lag)
    df[TARGET_NEXT] = g[TARGET_COL].shift(-1)
    feat = df.dropna(subset=FEATURE_COLS + [TARGET_NEXT]).reset_index(drop=True)
    return feat


def split_data(feat):
    """時間分割：終盤 VAL_MONTHS ヶ月を検証（未来）に。"""
    cutoff = feat["月"].max() - VAL_MONTHS
    train = feat[feat["月"] <= cutoff]
    val = feat[feat["月"] > cutoff]
    return (
        train[FEATURE_COLS], val[FEATURE_COLS],
        train[TARGET_NEXT], val[TARGET_NEXT],
        cutoff,
    )


def build_x_new(df, rep):
    """選んだ担当の「直近3ヶ月の活動」から、来月予測用の特徴量1行を作る。"""
    r = df[df["営業担当"] == rep].sort_values("月")
    last3 = r.tail(3)  # 2ヶ月前, 1ヶ月前, 当月
    vals = {}
    for act in ACT_COLS:
        series = list(last3[act])
        vals[f"{act}_当月"] = series[-1]
        vals[f"{act}_1ヶ月前"] = series[-2]
        vals[f"{act}_2ヶ月前"] = series[-3]
    return pd.DataFrame([vals])[FEATURE_COLS], last3


def build_model(n_estimators, max_depth):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )


def score(model, X, y):
    """決定係数 R²（1に近いほど当たっている）を返す。"""
    from sklearn.metrics import r2_score

    return float(r2_score(y, model.predict(X)))


# --- 🔒 編集コードの安全実行 ----------------------------------------------
def _exec_user(code, ns):
    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
        return None
    except Exception as e:  # noqa: BLE001
        return e


# --- コールバック（学習・推論） -------------------------------------------
def _do_fit():
    feat = build_features(load_data())
    X_train, X_val, y_train, y_val, _ = split_data(feat)
    model = build_model(st.session_state.n_estimators, st.session_state.max_depth)
    ns = {"model": model, "X_train": X_train, "y_train": y_train}

    err = _exec_user(st.session_state.fit_code, ns)
    if err is not None:
        st.session_state.fit_err = str(err)
        st.session_state.model = None
        return

    st.session_state.fit_err = None
    st.session_state.model = ns.get("model", model)
    st.session_state.train_score = score(st.session_state.model, X_train, y_train)
    st.session_state.val_score = score(st.session_state.model, X_val, y_val)
    st.session_state.prediction = None


def _do_predict():
    if st.session_state.get("model") is None:
        st.session_state.pred_err = "先に「学習」してください。"
        st.session_state.prediction = None
        return

    X_new, _ = build_x_new(load_data(), st.session_state.rep_select)
    ns = {"model": st.session_state.model, "X_new": X_new}

    err = _exec_user(st.session_state.predict_code, ns)
    if err is not None:
        st.session_state.pred_err = str(err)
        st.session_state.prediction = None
        return

    pred = ns.get("pred")
    if pred is None:
        st.session_state.pred_err = "pred に予測結果が入っていません。predict の書き方を見直してください。"
        st.session_state.prediction = None
        return

    st.session_state.pred_err = None
    st.session_state.prediction = float(pred[0])


# --- 画面 ------------------------------------------------------------------
def render_app(fit_snippet: str, predict_snippet: str, mode_label: str):
    st.set_page_config(page_title="数値AI ハンズオン｜来月の成約予測", page_icon="📈")
    st.title("📈 数値AIで「来月の成約数」を予測する")
    st.caption(f"先行活動（問い合わせ・架電・面談・提案）から翌月の成約を読む（モード：{mode_label}）")

    # セッション初期値
    st.session_state.setdefault("n_estimators", 20)
    st.session_state.setdefault("max_depth", 2)
    st.session_state.setdefault("fit_code", fit_snippet)
    st.session_state.setdefault("predict_code", predict_snippet)
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("prediction", None)

    # --- 🔒 やること一覧表 ---
    st.markdown(
        """
| | ステップ | あなたの操作 |
|---|---|---|
| 🔒 | Step1 データを見る | 見るだけ |
| 🟢 | Step2 fit を書く | ✍️ 編集セル → 🎓 学習ボタン |
| 🟢 | Step3 predict を書く | ✍️ 担当を選ぶ → 編集セル → 🔮 推論ボタン |
| 🟢 | Step4 パラメータを変える | 🎚️ スライダー → 学習し直す |

> **🔒＝触らないゾーン／🟢＝あなたが触るゾーン**。
"""
    )

    df = load_data()

    # --- 🔒 Step1: データを見る（生の時系列）---
    st.divider()
    st.subheader("🔒 Step1：データを見る（営業担当 × 月の時系列）")
    st.write("**営業の先行活動と、その月の成約数**。これを使い、AIに「翌月の成約数」を予測させます。")
    st.dataframe(df.head(12), width="stretch")
    feat = build_features(df)
    st.caption(
        f"生データ {len(df)} 行（{df['営業担当'].nunique()}担当 × {df['月'].max()}ヶ月）。"
        f"→ 当月・1ヶ月前・2ヶ月前の活動を特徴量化（{len(FEATURE_COLS)}列）し、"
        f"翌月の成約数を予測。学習用 {len(feat)} 行。"
    )

    # --- 🟢 Step4: パラメータ ---
    st.divider()
    st.subheader("🟢 Step4：パラメータ（変えて何度も学習しよう）")
    c1, c2 = st.columns(2)
    c1.slider("n_estimators（木の数＝学習回数）", 10, 500, key="n_estimators", step=10)
    c2.slider("max_depth（木の深さ）", 1, 10, key="max_depth")
    st.caption("大きくするほど訓練に強くフィット → でも検証（未来の月）で外れやすくなる（過学習）。")

    # --- 🟢 Step2: 学習（fit）---
    st.divider()
    st.subheader("🟢 Step2：学習する（fit を書く）")
    _, _, _, _, cutoff = split_data(feat)
    st.caption(f"検証は時間分割：{int(cutoff)}ヶ月目までで学習し、それ以降（未来）で検証します。")
    st.text_area("学習コード", key="fit_code", height=150)
    st.button("🎓 学習ボタン", type="primary", on_click=_do_fit)

    if st.session_state.get("fit_err"):
        st.error(f"⚠️ エラー: {st.session_state.fit_err}\n\nfit の書き方を見直してください。")
    elif st.session_state.get("model") is not None:
        st.success("学習できました！下のスコアと重要度を見てみましょう。")
        s1, s2 = st.columns(2)
        s1.metric("訓練スコア（train R²）", f"{st.session_state.train_score:.3f}")
        s2.metric("検証スコア（val R²）", f"{st.session_state.val_score:.3f}")
        gap = st.session_state.train_score - st.session_state.val_score
        if gap > 0.2:
            st.warning(
                f"訓練 {st.session_state.train_score:.3f} に対し検証 {st.session_state.val_score:.3f}。"
                "差が開いています＝**過学習**ぎみ。パラメータを下げてみましょう。"
            )
        else:
            st.info("訓練と検証のスコアが近い＝バランス良好です。")

        # --- 🔒 特徴量の重要度 ---
        importance = pd.Series(
            st.session_state.model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        st.write("**どの活動が翌月の成約に効いた？（特徴量の重要度）**")
        st.bar_chart(importance)

    # --- 🟢 Step3: 推論（predict）---
    st.divider()
    st.subheader("🟢 Step3：推論する（predict を書く）")
    reps = sorted(df["営業担当"].unique())
    st.selectbox("予測する営業担当を選ぶ", reps, key="rep_select")
    _, last3 = build_x_new(df, st.session_state.rep_select)
    st.caption(f"{st.session_state.rep_select} の直近3ヶ月の活動（これを使って来月を予測）:")
    st.dataframe(last3[["月", *ACT_COLS]], width="stretch", hide_index=True)

    st.text_area("推論コード", key="predict_code", height=130)
    st.button("🔮 推論ボタン", type="primary", on_click=_do_predict)

    if st.session_state.get("pred_err"):
        st.error(f"⚠️ {st.session_state.pred_err}")
    elif st.session_state.get("prediction") is not None:
        st.metric(f"{st.session_state.rep_select} の翌月の予測成約数",
                  f"{st.session_state.prediction:.2f} 件")
        st.caption("担当を変えたり、パラメータを変えて、予測がどう動くか試してみましょう。")
