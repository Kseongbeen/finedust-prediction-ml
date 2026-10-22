# ============================================================
# 📊 Gradient Boosting 하이퍼파라미터 튜닝 (산불 제외)
# ============================================================
# 새로운 셀에 붙여넣기
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
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

train_x_final = preprocess_with_hour(train_x)
test_x_final = preprocess_with_hour(test_x)

if 'time' in train_y.columns: train_y = train_y.drop(columns=['time'])
if 'time' in test_y.columns: test_y = test_y.drop(columns=['time'])

common_cols = sorted(list(set(train_x_final.columns) & set(test_x_final.columns)))
train_x_final = train_x_final[common_cols].ffill().bfill()
test_x_final = test_x_final[common_cols].ffill().bfill()

print(f'✅ 학습 변수 ({len(common_cols)}개): {common_cols}')

# 3. 산불 기간(3/4~3/6) 제외
test_x_raw = pd.read_csv('test_x.csv')
if 'time' in test_x_raw.columns:
    test_idx = pd.to_datetime(test_x_raw['time'])
    fire_mask = (test_idx >= '2022-03-04') & (test_idx < '2022-03-07')
    test_x_final = test_x_final[~fire_mask.values]
    test_y = test_y[~fire_mask.values]
    print(f'🔥 산불 제외: {fire_mask.sum()}개 제거 → 남은 테스트: {len(test_x_final)}개')

# 4. 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x_final)
test_x_scaled = scaler.transform(test_x_final)

# 5. 기본 모델 성능 (비교용)
base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
base_model.fit(train_x_scaled, train_y.values.ravel())
base_pred = base_model.predict(test_x_scaled)
base_r2 = r2_score(test_y, base_pred)
print(f'\n📊 기본 GB 성능: R² = {base_r2:.6f}')

# 6. 하이퍼파라미터 탐색 범위
param_dist = {
    'n_estimators': [200, 300, 500, 700, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 5, 8],
    'subsample': [0.7, 0.8, 0.85, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', 0.5, 0.8, None],
}

print('\n🔍 RandomizedSearchCV 탐색 시작...')
print('   (약 15~20분 소요, 밥 먹고 오세요 🍚)')

gb = GradientBoostingRegressor(random_state=42)
search = RandomizedSearchCV(
    gb,
    param_distributions=param_dist,
    n_iter=200,          # 200개 조합 탐색 (충분히 오래 걸림)
    scoring='r2',
    cv=3,                # 3-fold CV
    random_state=42,
    n_jobs=-1,           # 모든 CPU 사용
    verbose=1
)
search.fit(train_x_scaled, train_y.values.ravel())

print(f'\n✅ 최적 파라미터:')
for k, v in search.best_params_.items():
    print(f'   {k}: {v}')
print(f'\n📈 CV 최고 R²: {search.best_score_:.6f}')

# 7. 최적 모델로 테스트 평가
best_model = search.best_estimator_
best_pred = best_model.predict(test_x_scaled)

best_r2 = r2_score(test_y, best_pred)
best_mae = mean_absolute_error(test_y, best_pred)
best_rmse = np.sqrt(mean_squared_error(test_y, best_pred))

print(f'\n📈 최적 GB 테스트 성능 (산불 제외):')
print(f'   R²   = {best_r2:.6f}  (기본: {base_r2:.6f}, 개선: {best_r2 - base_r2:+.6f})')
print(f'   MAE  = {best_mae:.4f}')
print(f'   RMSE = {best_rmse:.4f}')

# 8. 변수 중요도 시각화
importances = best_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': common_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 산점도
axes[0].scatter(test_y, best_pred, alpha=0.3, s=10, color='#2ecc71')
mx = max(test_y.max().values[0], best_pred.max())
axes[0].plot([0, mx], [0, mx], 'r--', linewidth=2)
axes[0].set_xlabel('실제 PM10')
axes[0].set_ylabel('예측 PM10')
axes[0].set_title(f'최적 GB (R²={best_r2:.4f})', fontsize=13, fontweight='bold')

# 변수 중요도 (Top 15)
top15 = feat_imp.head(15)
sns.barplot(x='Importance', y='Feature', data=top15, palette='viridis', ax=axes[1])
axes[1].set_title('변수 중요도 (Top 15)', fontsize=13, fontweight='bold')

plt.suptitle('🔥 GB 하이퍼파라미터 튜닝 결과 (산불 제외)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print('\n✅ 튜닝 완료!')
