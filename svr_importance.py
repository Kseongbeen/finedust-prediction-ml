# ============================================================
# 📊 SVR (PM 미포함 + PCA 99%) + 변수 중요도
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import platform

if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
train_x = pd.read_csv('train_x.csv')
train_y = pd.read_csv('train_y.csv')
test_x = pd.read_csv('test_x.csv')
test_y = pd.read_csv('test_y.csv')

# 2. 전처리
def preprocess_with_hour(df):
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        if 'hour' not in df.columns:
            df['hour'] = df['time'].dt.hour
        df = df.drop(columns=['time'])
    return df

train_x = preprocess_with_hour(train_x)
test_x = preprocess_with_hour(test_x)

if 'time' in train_y.columns: train_y = train_y.drop(columns=['time'])
if 'time' in test_y.columns: test_y = test_y.drop(columns=['time'])

# 3. PM 변수 제거
pm_cols = [col for col in train_x.columns if 'PM' in col]
train_x = train_x.drop(columns=pm_cols)
test_x = test_x.drop(columns=pm_cols)

common_cols = sorted(list(set(train_x.columns) & set(test_x.columns)))
train_x = train_x[common_cols].ffill().bfill()
test_x = test_x[common_cols].ffill().bfill()

print(f'✅ PM 제거 후 변수 ({len(common_cols)}개): {common_cols}')

# 4. 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 5. PCA (99% 분산 유지)
pca = PCA(n_components=0.99, random_state=42)
train_x_pca = pca.fit_transform(train_x_scaled)
test_x_pca = pca.transform(test_x_scaled)

print(f'📊 PCA: {len(common_cols)}개 → {pca.n_components_}개 (분산 {pca.explained_variance_ratio_.sum()*100:.1f}% 유지)')

# 6. SVR 학습 (튜닝된 파라미터)
print('\n🔧 SVR 학습 중 (C=70, epsilon=0.12, gamma=0.004)...')
svr_model = SVR(kernel='rbf', C=70, epsilon=0.12, gamma=0.004)
svr_model.fit(train_x_pca, train_y.values.ravel())

# 7. 예측 & 평가
pred_y = svr_model.predict(test_x_pca)

r2 = r2_score(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
rmse = np.sqrt(mean_squared_error(test_y, pred_y))

print(f'\n📈 SVR 성능 (PM 미포함 + PCA 99%):')
print(f'   R²   = {r2:.6f}')
print(f'   MAE  = {mae:.4f}')
print(f'   RMSE = {rmse:.4f}')

# 8. Permutation Importance (PCA 전 원본 변수 기준)
print('\n🔍 Permutation Importance 계산 중... (약 1~2분)')

# PCA 없이 원본 스케일 데이터로 SVR 다시 학습 (변수 중요도용)
svr_orig = SVR(kernel='rbf', C=70, epsilon=0.12, gamma=0.004)
svr_orig.fit(train_x_scaled, train_y.values.ravel())

perm_result = permutation_importance(
    svr_orig,
    test_x_scaled,
    test_y.values.ravel(),
    n_repeats=5,         # 5번 반복 (빠른 확인용)
    random_state=42,
    scoring='r2'
)

perm_imp = pd.DataFrame({
    'Feature': common_cols,
    'Importance': perm_result.importances_mean,
    'Std': perm_result.importances_std
}).sort_values('Importance', ascending=False)

print('\n📊 SVR 변수 중요도:')
print(perm_imp.to_string(index=False))

# 9. 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 실제 vs 예측
axes[0].scatter(test_y, pred_y, alpha=0.3, s=10, color='#e67e22')
mx = max(test_y.max().values[0], pred_y.max())
axes[0].plot([0, mx], [0, mx], 'r--', linewidth=2)
axes[0].set_xlabel('실제 PM10')
axes[0].set_ylabel('예측 PM10')
axes[0].set_title(f'SVR 실제 vs 예측 (R²={r2:.4f})', fontsize=13, fontweight='bold')

# 변수 중요도 Top 15
top15 = perm_imp.head(15)
colors = ['#e74c3c' if v > 0 else '#3498db' for v in top15['Importance']]
axes[1].barh(top15['Feature'], top15['Importance'], xerr=top15['Std'], color=colors, alpha=0.8)
axes[1].set_xlabel('중요도 (R² 감소량)')
axes[1].set_title('SVR 변수 중요도 Top 15 (PM 미포함)', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print('\n✅ 완료!')
