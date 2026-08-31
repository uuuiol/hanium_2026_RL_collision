"""
top10_episodes.py

멀티에이전트 PPO 학습 결과 CSV(multi_ppo_results.csv 등)에서
avg_score(평균 점수)가 가장 높았던 에피소드 상위 10개를 뽑아서 보여줍니다.

사용법: 이 파일을 결과 CSV와 같은 폴더에 놓고
    python top10_episodes.py
라고 실행하면 됩니다. 다른 CSV를 보고 싶으면 아래 CSV_PATH만 바꾸세요
(예: "multi_ppo_colreg_results.csv").
"""
import csv

CSV_PATH = "multi_ppo_results.csv"   # 보고 싶은 결과 파일 이름으로 바꿔도 됩니다
TOP_N = 10

rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# avg_score 기준으로 높은 순 정렬
rows_sorted = sorted(rows, key=lambda r: float(r["avg_score"]), reverse=True)
top_rows = rows_sorted[:TOP_N]

print(f"{CSV_PATH} 총 {len(rows)}개 에피소드 중 점수 상위 {TOP_N}개\n")
header = f"{'순위':<4}{'에피소드':<10}{'평균score':<12}{'목표도달':<10}{'충돌':<8}{'COLREG준수':<12}"
print(header)
print("-" * len(header))

for rank, row in enumerate(top_rows, start=1):
    colreg = row.get("colreg_compliance_rate", "")
    colreg_str = f"{float(colreg)*100:.0f}%" if colreg not in ("", None) else "N/A"
    print(f"{rank:<4}{row['world_episode']:<10}{float(row['avg_score']):<12.2f}"
          f"{row['goal_count']:<10}{row['collision_count']:<8}{colreg_str:<12}")

# 상위 10개만 별도 CSV로도 저장
out_path = CSV_PATH.replace(".csv", "_top10.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(top_rows)
print(f"\n상위 {TOP_N}개를 별도 파일로도 저장했습니다: {out_path}")
