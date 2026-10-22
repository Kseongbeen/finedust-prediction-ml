# ============================================================
# 📊 VIF > 10 반복 제거 + 피어슨 |r| > 0.9 제거 코드
# ============================================================
# 노트북 맨 아래 셀에 붙여넣기 (독립 실행 가능)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──
train_x = pd.read_csv('train_x.csv', index_col=0)

# ── 파생변수 제외 ──
drop_from_analysis = ['hour', 'PM10_lag24', 'PM10_lag1', 'PM10_lag']
analysis_cols = [c for c in train_x.select_dtypes(include=[np.number]).columns
                 if c not in drop_from_analysis]
df = train_x[analysis_cols].dropna()
print(f'✅ 시작 변수 ({len(analysis_cols)}개): {analysis_cols}\n')

# ── VIF 계산 함수 ──
def calc_vif(df):
    X = df.values.astype(float)
    vifs = {}
    for i, col in enumerate(df.columns):
        y = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        X_rest = np.hstack([np.ones((X_rest.shape[0], 1)), X_rest])
        beta = np.linalg.lstsq(X_rest, y, rcond=None)[0]
        y_pred = X_rest @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        vifs[col] = round(1 / (1 - r_sq) if r_sq < 1 else float('inf'), 2)
    return vifs


# ==========================================
# [1] 피어슨 |r| > 0.9 쌍에서 VIF 높은 쪽 제거
# ==========================================
print('=' * 60)
print('📊 [1단계] 피어슨 |r| > 0.9 → VIF 높은 쪽 제거')
print('=' * 60)

corr = df.corr()
corr_removed = []

while True:
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                pairs.append((corr.columns[i], corr.columns[j], abs(corr.iloc[i, j])))
    
    if not pairs:
        break
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    c1, c2, r_val = pairs[0]
    
    vifs_full = calc_vif(df)
    v1, v2 = vifs_full[c1], vifs_full[c2]
    
    drop_col = c1 if v1 >= v2 else c2
    keep_col = c2 if drop_col == c1 else c1
    
    print(f'\n   {c1} ↔ {c2}  |r| = {r_val:.4f}')
    print(f'   → {c1} VIF={v1}  /  {c2} VIF={v2}')
    print(f'   → {drop_col} 제거 (VIF 더 높음), {keep_col} 유지 ✅')
    
    corr_removed.append(drop_col)
    df = df.drop(columns=[drop_col])
    corr = df.corr()

print(f'\n🔴 상관계수로 제거된 변수 ({len(corr_removed)}개): {corr_removed}')


# ==========================================
# [2] VIF > 10 반복 제거
# ==========================================
print('\n' + '=' * 60)
print('📊 [2단계] VIF > 10 반복 제거 (가장 높은 것부터)')
print('=' * 60)

vif_removed = []
step = 0

while True:
    vifs = calc_vif(df)
    max_col = max(vifs, key=vifs.get)
    max_vif = vifs[max_col]
    
    if max_vif <= 10:
        print(f'\n   ✅ 모든 변수 VIF ≤ 10 달성! (반복 {step}회)')
        break
    
    step += 1
    print(f'   [{step}회] {max_col} 제거 (VIF = {max_vif:,.1f})')
    vif_removed.append((max_col, max_vif))
    df = df.drop(columns=[max_col])

print(f'\n🔴 VIF로 제거된 변수 ({len(vif_removed)}개):')
for col, v in vif_removed:
    print(f'   - {col} (VIF = {v:,.1f})')


# ==========================================
# [3] 최종 결과
# ==========================================
print('\n' + '=' * 60)
print('📋 [최종 결과]')
print('=' * 60)

all_removed = corr_removed + [c for c, v in vif_removed]
final_cols = df.columns.tolist()

final_vifs = calc_vif(df)
vif_final = pd.DataFrame({
    '변수명': list(final_vifs.keys()),
    'VIF': list(final_vifs.values())
}).sort_values('VIF', ascending=False).reset_index(drop=True)

print(f'\n✅ 남은 변수 ({len(final_cols)}개):')
print(vif_final.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, max(5, len(final_cols)*0.4)))
colors = ['#e74c3c' if v > 10 else '#2ecc71' for v in vif_final['VIF']]
bars = ax.barh(vif_final['변수명'], vif_final['VIF'], color=colors)
ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 기준선')
ax.set_xlabel('VIF', fontsize=12)
ax.set_title('✅ 최종 변수 VIF (모두 10 이하)', fontsize=14, fontweight='bold')
for bar, v in zip(bars, vif_final['VIF']):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}', va='center', fontsize=9)
ax.legend(fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
final_corr = df.corr()
mask = np.triu(np.ones_like(final_corr, dtype=bool))
sns.heatmap(final_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax)
ax.set_title('✅ 최종 피어슨 상관계수 히트맵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print('\n' + '=' * 60)
print('📋 제거된 변수 전체 목록')
print('=' * 60)
print(f'\n[상관계수 |r| > 0.9 제거] ({len(corr_removed)}개):')
for c in corr_removed:
    print(f'   ❌ {c}')
print(f'\n[VIF > 10 반복 제거] ({len(vif_removed)}개):')
for c, v in vif_removed:
    print(f'   ❌ {c} (VIF={v:,.1f})')
print(f'\n총 제거: {len(all_removed)}개  /  남은 변수: {len(final_cols)}개')
print(f'남은 변수 목록: {final_cols}')
