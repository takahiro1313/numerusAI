"""数値AIハンズオン用のダミー時系列CSVを生成する（SFDCイメージ）。

営業担当 × 月 のパネルデータ。各月の「先行活動」から、AIが
「翌月の成約数」を予測できるように、活動→成約にリードタイム（ラグ）を持たせる。
結果系（契約数）は入れない（自明回避）。実データは一切使わない（要件 §4）。

生データ列: 営業担当, 月, 問い合わせ数, 架電数, 面談数, 案件提案数, 成約数
（ラグ特徴量づくりと「翌月成約」へのズラしは ai_core.py 側で行う＝🔒前処理）

実行:
    python make_dummy.py
出力:
    data/activities.csv
"""

import os
import numpy as np
import pandas as pd

N_REPS = 30          # 営業担当の人数
N_MONTHS = 30        # 月数（1..30）
SEED = 42

# 先行活動（結果系=契約数は含めない）
ACT_COLS = ["問い合わせ数", "架電数", "面談数", "案件提案数"]
TARGET_COL = "成約数"


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    # 各活動の平均・ばらつき（担当ごとに独立に baseline を引く＝活動間の共線性を避ける）
    ACT_MEAN_SD = {
        "問い合わせ数": (45, 10),
        "架電数": (85, 20),
        "面談数": (28, 7),
        "案件提案数": (9, 3),
    }

    for rep in range(1, N_REPS + 1):
        # 担当ごとに、活動ごと独立のベース水準（互いに相関しない）
        base = {c: float(max(1.0, rng.normal(m, s))) for c, (m, s) in ACT_MEAN_SD.items()}
        history = []  # 各月の dict(活動量)

        for month in range(1, N_MONTHS + 1):
            act = {c: int(max(0, rng.poisson(base[c]))) for c in ACT_COLS}
            history.append(act)

            # 成約数(今月) = 「過去の」先行活動だけから決まる（リードタイムあり）。
            # ・案件提案（1〜2ヶ月前）が最も効く、面談がそれを補助。
            # ・問い合わせは弱く効く。架電は“あえて効かない指標”（量より質の示唆）。
            # get(-2)=先月, get(-3)=先々月, get(-4)=3ヶ月前
            def get(idx, col):
                return history[idx][col] if -len(history) <= idx < len(history) else 0

            lead = (
                0.16 * get(-2, "案件提案数")     # 先月の提案が最重要
                + 0.12 * get(-3, "案件提案数")   # 先々月の提案も効く
                + 0.06 * get(-2, "面談数")       # 先月の面談
                + 0.03 * get(-3, "面談数")
                + 0.012 * get(-4, "問い合わせ数")  # 入口は弱く効く
                # 架電数は意図的に係数ゼロ（効かない指標）
            )
            noise = rng.normal(0, 1.0)
            seiyaku = int(np.clip(round(lead + noise), 0, None))

            rows.append(
                {
                    "営業担当": f"担当{rep:02d}",
                    "月": month,
                    **act,
                    TARGET_COL: seiyaku,
                }
            )

    df = pd.DataFrame(rows, columns=["営業担当", "月", *ACT_COLS, TARGET_COL])

    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "activities.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"生成しました: {out}（{len(df)}行＝{N_REPS}担当 × {N_MONTHS}ヶ月）")
    print(df.head(8))
    print("\n成約数の分布:", df[TARGET_COL].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
