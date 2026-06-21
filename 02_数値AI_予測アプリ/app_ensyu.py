"""数値AI（活動量→Ad数予測）ハンズオン — 演習用（参加者に配布）。

fit / predict が空欄（___）。
- 学習：model.fit(___, ___) に X_train, y_train を入れる
- 推論：model.predict(___) に X_new を入れる
（正解は app_seikai.py）
"""

from ai_core import render_app

# 🟢 学習（fit）の編集セル初期値（空欄）
FIT_SNIPPET = '''# 🟢 ===== ここを編集 =====
# 活動量 X_train と 正解 y_train を渡してモデルを学習させよう
model.fit(___, ___)        # ← ___ に X_train, y_train を入れる
# 🟢 ===== 編集はここまで =====
'''

# 🟢 推論（predict）の編集セル初期値（空欄）
PREDICT_SNIPPET = '''# 🟢 ===== ここを編集 =====
# 学習済みモデルに新しい活動量 X_new を渡して Ad数 を予測しよう
pred = model.predict(___)  # ← ___ に X_new を入れる
# 🟢 ===== 編集はここまで =====
'''

render_app(FIT_SNIPPET, PREDICT_SNIPPET, mode_label="演習用")
