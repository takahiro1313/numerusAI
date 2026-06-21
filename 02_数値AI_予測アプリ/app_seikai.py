"""数値AI（活動量→Ad数予測）ハンズオン — 正解用（講師リファレンス）。

演習用（app_ensyu.py）の ___ を正解で埋めたもの。
そのまま学習ボタン・推論ボタンを押せば動く。
"""

from ai_core import render_app

# 🟢 学習（fit）の編集セル初期値（記入済み）
FIT_SNIPPET = '''# 🟢 ===== ここを編集 =====
model.fit(X_train, y_train)
# 🟢 ===== 編集はここまで =====
'''

# 🟢 推論（predict）の編集セル初期値（記入済み）
PREDICT_SNIPPET = '''# 🟢 ===== ここを編集 =====
pred = model.predict(X_new)
# 🟢 ===== 編集はここまで =====
'''

render_app(FIT_SNIPPET, PREDICT_SNIPPET, mode_label="正解用")
