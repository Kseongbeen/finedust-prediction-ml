# ============================================================
# 📊 선형 회귀: VIF/상관계수 정리 후 남은 변수로 PM10 예측
# ============================================================
# 새로운 셀에 붙여넣기 (vif_correlation_check 실행 후 사용)
# final_cols 변수가 필요합니다
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 데이터 로드
train_x = pd.read_csv('train_x.csv', index_col=0)
train_y = pd.read_csv('train_y.csv', index_col=0)
test_x = pd.read_csv('test_x.csv', index_col=0)
test_y = pd.read_csv('test_y.csv', index_col=0)

# VIF/상관계수 분석에서 살아남은 변수 + lag 변수
lag_cols = [c for c in ['PM10_lag24', 'PM10_lag1'] if c in train_x.columns and c in test_x.columns]
use_cols = final_cols + lag_cols
use_cols = [c for c in use_cols if c in train_x.columns and c in test_x.columns]
print(f'✅ 학습 변수 ({len(use_cols)}개): {use_cols}\n')

# 학습/테스트 데이터 준비
X_train = train_x[use_cols].dropna()
y_train = train_y.loc[X_train.index]
X_test = test_x[use_cols].dropna()
y_test = test_y.loc[X_test.index]

# ── 선형 회귀 학습 (2021) & 평가 (2022) ──
model = LinearRegression()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, pred_train)
test_r2 = r2_score(y_test, pred_test)
test_mae = mean_absolute_error(y_test, pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

print(f'📈 [전체 2022] 성능:')
print(f'   Train R² = {train_r2:.4f}')
print(f'   Test  R² = {test_r2:.4f}')
print(f'   Test MAE = {test_mae:.2f}')
print(f'   Test RMSE = {test_rmse:.2f}')

# 회귀 계수
coef_df = pd.DataFrame({
    '변수': use_cols,
    '회귀계수': model.coef_.flatten()
}).sort_values('회귀계수', ascending=False, key=abs).reset_index(drop=True)
print(f'\n📊 회귀 계수 (영향력 순):')
print(coef_df.to_string(index=False))

# ── 산불 기간(03/04~03/06) 제외 재평가 ──
print('\n' + '-' * 50)
print('🔥 울진삼척 산불(2022-03-04 ~ 03-06) 제외 재평가')
print('-' * 50)

test_idx = pd.to_datetime(X_test.index)
fire_mask = (test_idx >= '2022-03-04') & (test_idx < '2022-03-07')

X_test_clean = X_test[~fire_mask]
y_test_clean = y_test.loc[X_test_clean.index]
pred_clean = model.predict(X_test_clean)

clean_r2 = r2_score(y_test_clean, pred_clean)
clean_mae = mean_absolute_error(y_test_clean, pred_clean)
clean_rmse = np.sqrt(mean_squared_error(y_test_clean, pred_clean))

print(f'\n   제거 데이터: {fire_mask.sum()}개  /  남은 데이터: {len(X_test_clean)}개')
print(f'\n📈 성능 비교:')
print(f'   {"":15s} {"전체 2022":>10s}  {"산불 제외":>10s}')
print(f'   {"R²":15s} {test_r2:>10.4f}  {clean_r2:>10.4f}')
print(f'   {"MAE":15s} {test_mae:>10.2f}  {clean_mae:>10.2f}')
print(f'   {"RMSE":15s} {test_rmse:>10.2f}  {clean_rmse:>10.2f}')

# ── 비교 시각화 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, pred_test, alpha=0.3, s=10, color='#3498db')
mx = max(y_test.max().values[0], pred_test.max())
axes[0].plot([0, mx], [0, mx], 'r--', linewidth=2)
axes[0].set_xlabel('실제 PM10')
axes[0].set_ylabel('예측 PM10')
axes[0].set_title(f'전체 2022 (R²={test_r2:.4f})', fontsize=13, fontweight='bold')

axes[1].scatter(y_test_clean, pred_clean, alpha=0.3, s=10, color='#e67e22')
mx2 = max(y_test_clean.max().values[0], pred_clean.max())
axes[1].plot([0, mx2], [0, mx2], 'r--', linewidth=2)
axes[1].set_xlabel('실제 PM10')
axes[1].set_ylabel('예측 PM10')
axes[1].set_title(f'산불 제외 (R²={clean_r2:.4f})', fontsize=13, fontweight='bold')

plt.suptitle('🔥 산불 제외 전후 비교', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print('\n✅ 완료!')
