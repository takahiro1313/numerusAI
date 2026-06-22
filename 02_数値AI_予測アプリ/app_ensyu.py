"""数値AI ハンズオン — 演習用（参加者に配布）。

「来月の成約数」を予測するAIを、アプリ内の編集セルに fit/predict を書いて動かす。
このアプリは **Streamlit Community Cloud にデプロイして使う**（学習・推論の計算はサーバ側。
役員PCに負荷をかけない）。参加者は配布URLを開き、🟢の編集セルだけを触る。

▼ ローカル確認:
    pip install -r requirements.txt
    python make_dummy.py
    streamlit run app_ensyu.py
正解は app_seikai.py。
"""

import os
import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score

# =========================================================================
# 🔒 裏方（データ下ごしらえ・モデル・安全実行）。さわらない。
# =========================================================================
ACT_COLS = ["問い合わせ数", "架電数", "面談数", "案件提案数"]
LAGS = [("当月", 0), ("1ヶ月前", 1), ("2ヶ月前", 2)]
FEATURE_COLS = [f"{a}_{label}" for a in ACT_COLS for (label, _) in LAGS]  # 手がかり12列
TARGET_NEXT = "翌月成約数"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "activities.csv")
PRED_PATH = os.path.join(os.path.dirname(__file__), "data", "予測結果.csv")  # まーぴーが読む
VAL_MONTHS = 3  # 終盤3ヶ月を「未来（検証）」に回す

# 編集セルを安全に実行するための最小ビルトイン（import/IO は渡さない）
SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "len": len, "range": range, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "print": print, "round": round,
}


def load_data():
    return pd.read_csv(DATA_PATH)


def build_features(df):
    """当月/1ヶ月前/2ヶ月前の活動を手がかりにし、翌月の成約数を答えにする。"""
    df = df.sort_values(["営業担当", "月"]).copy()
    g = df.groupby("営業担当", group_keys=False)
    for act in ACT_COLS:
        for (label, lag) in LAGS:
            df[f"{act}_{label}"] = g[act].shift(lag)
    df[TARGET_NEXT] = g["成約数"].shift(-1)
    return df.dropna(subset=FEATURE_COLS + [TARGET_NEXT]).reset_index(drop=True)


def split_data(feat):
    """過去で学習し、未来（終盤の月）で答え合わせ（時間で分ける）。"""
    cutoff = feat["月"].max() - VAL_MONTHS
    tr, va = feat[feat["月"] <= cutoff], feat[feat["月"] > cutoff]
    return tr[FEATURE_COLS], va[FEATURE_COLS], tr[TARGET_NEXT], va[TARGET_NEXT], cutoff


def build_x_new(df, rep):
    """選んだ担当の“直近3ヶ月の活動”から、来月予測用の手がかり1行を作る。"""
    last3 = df[df["営業担当"] == rep].sort_values("月").tail(3)
    vals = {}
    for act in ACT_COLS:
        s = list(last3[act])
        vals[f"{act}_当月"], vals[f"{act}_1ヶ月前"], vals[f"{act}_2ヶ月前"] = s[-1], s[-2], s[-3]
    return pd.DataFrame([vals])[FEATURE_COLS], last3


def build_model():
    """🟢で選んだ種類とパラメータからモデルを作る（学習はまだしない）。"""
    if st.session_state.get("model_kind", "木").startswith("直線"):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=int(st.session_state.n_estimators),
        max_depth=int(st.session_state.max_depth),
        learning_rate=0.1, random_state=42, verbosity=0,
    )


def score(model, X, y):
    return float(r2_score(y, model.predict(X)))


def save_all_predictions(model, df):
    """学習済みモデルで全担当の来月を予測し、まーぴー用CSVに保存（🔒 連結）。"""
    rows = []
    for rep in sorted(df["営業担当"].unique()):
        x_new, _ = build_x_new(df, rep)
        rows.append({"営業担当": rep, "予測_翌月成約数": round(float(model.predict(x_new)[0]), 1)})
    pd.DataFrame(rows).to_csv(PRED_PATH, index=False, encoding="utf-8-sig")


def _exec_user(code, ns):
    """編集セルのコードを限定名前空間で実行。成功=None／失敗=例外。"""
    try:
        exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
        return None
    except Exception as e:  # noqa: BLE001  事故防止のため全例外を画面表示
        return e


# --- コールバック（学習・推論） ------------------------------------------
def _do_fit():
    feat = build_features(load_data())
    X_train, X_val, y_train, y_val, _ = split_data(feat)
    model = build_model()
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
    save_all_predictions(st.session_state.model, load_data())  # 🔗 まーぴー連結用


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


# =========================================================================
# 画面（🔒 描画。🟢 は編集セルとスライダー）
# =========================================================================
def render_app(fit_snippet: str, predict_snippet: str, mode_label: str):
    st.set_page_config(page_title="数値AI｜来月の成約予測", page_icon="📈")
    st.title("📈 来月の成約数を予測するAI")
    st.caption(f"先行活動から翌月の成約を読む（モード：{mode_label}）")

    st.session_state.setdefault("model_kind", "木のモデル（XGBoost）")
    st.session_state.setdefault("n_estimators", 40)
    st.session_state.setdefault("max_depth", 2)
    st.session_state.setdefault("fit_code", fit_snippet)
    st.session_state.setdefault("predict_code", predict_snippet)
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("prediction", None)

    st.markdown(
        """
### このハンズオンでやること
IT編では「**人がルール(if)を書いて**」案内を出しました。今回は「**AIがデータからルールを学んで**」予測します。

| | ステップ | あなたの操作 |
|---|---|---|
| 🔒 | Step1 データを見る | 見るだけ（手がかりX／答えy） |
| 🟢 | Step2 モデルを選ぶ | 種類とパラメータを選ぶ |
| 🟢 | Step3 学習する（fit） | ✍️ 編集セル → 🎓 学習 |
| 🟢 | Step4 推論する（predict） | 担当を選ぶ → ✍️ 編集セル → 🔮 推論 |

> 🟢＝あなたが触るゾーン／🔒＝裏方（さわらない）。
"""
    )

    df = load_data()
    feat = build_features(df)

    # --- 🔒 Step1: データを見る ---
    st.divider()
    st.subheader("🔒 Step1：データを見る（営業担当 × 月）")
    st.dataframe(df.head(12), width="stretch")
    _, _, _, _, cutoff = split_data(feat)
    st.caption(
        f"**手がかり(X)＝活動4種**（{', '.join(ACT_COLS)}）の当月・1・2ヶ月前。"
        f" **答え(y)＝翌月の成約数**。学習用 {len(feat)} 行（過去{int(cutoff)}ヶ月で学習→残りで検証）。"
    )

    # --- 🟢 Step2: モデルを選ぶ ---
    st.divider()
    st.subheader("🟢 Step2：モデルを選ぶ")
    st.radio("モデルの種類", ["木のモデル（XGBoost）", "直線モデル（線形回帰）"], key="model_kind")
    if st.session_state.model_kind.startswith("木"):
        c1, c2 = st.columns(2)
        c1.slider("n_estimators（木の数）", 10, 500, key="n_estimators", step=10)
        c2.slider("max_depth（木の深さ）", 1, 10, key="max_depth")
        st.caption("数字を大きくするほど訓練に強くフィット → でも検証（未来の月）で外れやすい（過学習）。")

    # --- 🟢 Step3: 学習（fit）---
    st.divider()
    st.subheader("🟢 Step3：学習する（fit を書く）")
    st.text_area("学習コード", key="fit_code", height=140)
    st.button("🎓 学習", type="primary", on_click=_do_fit)

    if st.session_state.get("fit_err"):
        st.error(f"⚠️ エラー: {st.session_state.fit_err}\n\nfit の書き方を見直してください。")
    elif st.session_state.get("model") is not None:
        st.success("学習できました！スコアを見てみましょう。")
        s1, s2 = st.columns(2)
        s1.metric("訓練スコア（train R²）", f"{st.session_state.train_score:.3f}")
        s2.metric("検証スコア（val R²）", f"{st.session_state.val_score:.3f}")
        if st.session_state.train_score - st.session_state.val_score > 0.2:
            st.warning("訓練◎なのに検証✕＝**過学習**ぎみ。Step2の数字を下げてみましょう。")
        else:
            st.info("訓練と検証のスコアが近い＝バランス良好です。")
        if hasattr(st.session_state.model, "feature_importances_"):
            imp = pd.Series(
                st.session_state.model.feature_importances_, index=FEATURE_COLS
            ).sort_values(ascending=False)
            st.write("**どの活動が翌月の成約に効いた？（重要度）**")
            st.bar_chart(imp)

    # --- 🟢 Step4: 推論（predict）---
    st.divider()
    st.subheader("🟢 Step4：推論する（predict を書く）")
    st.selectbox("予測する営業担当を選ぶ", sorted(df["営業担当"].unique()), key="rep_select")
    _, last3 = build_x_new(df, st.session_state.rep_select)
    st.caption(f"{st.session_state.rep_select} の直近3ヶ月の活動（これを手がかりに来月を予測）:")
    st.dataframe(last3[["月", *ACT_COLS]], width="stretch", hide_index=True)
    st.text_area("推論コード", key="predict_code", height=120)
    st.button("🔮 推論", type="primary", on_click=_do_predict)

    if st.session_state.get("pred_err"):
        st.error(f"⚠️ {st.session_state.pred_err}")
    elif st.session_state.get("prediction") is not None:
        st.metric(f"🔮 {st.session_state.rep_select} の来月の予測成約数",
                  f"{st.session_state.prediction:.1f} 件")
        st.caption("担当やパラメータを変えて、予測がどう動くか試してみましょう。")


# 🟢 学習（fit）の編集セル初期値（演習用・空欄）
FIT_SNIPPET = '''# 🟢 ===== ここを編集 =====
# 手がかり X_train を見せて、答え y_train を当てさせるよう学習させよう
model.fit(___, ___)        # ← ___ に X_train, y_train を入れる
# 🟢 ===== 編集はここまで =====
'''

# 🟢 推論（predict）の編集セル初期値（演習用・空欄）
PREDICT_SNIPPET = '''# 🟢 ===== ここを編集 =====
# 学習済みモデルに、選んだ担当の直近活動 X_new を渡して来月を予測しよう
pred = model.predict(___)  # ← ___ に X_new を入れる
# 🟢 ===== 編集はここまで =====
'''

render_app(FIT_SNIPPET, PREDICT_SNIPPET, mode_label="演習用")
