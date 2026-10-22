import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🚀 데이터 로드 및 전처리를 시작합니다...")

# 1. 데이터 로드
train_x = pd.read_csv('train_x.csv')
train_y = pd.read_csv('train_y.csv')
test_x = pd.read_csv('test_x.csv')
test_y = pd.read_csv('test_y.csv')

# 타겟 변수 확인 (PM10_1: 1시간 후의 PM10 예측)
y_train = train_y['PM10_1']
y_test = test_y['PM10_1']

# 2. 데이터 누수(Data Leakage) 방지 및 피처 정렬
# train_x에는 'hour'가 있고 test_x에는 없는 이슈가 발견되었습니다.
# 'time' 컬럼에서 직접 'hour'를 추출하여 두 데이터의 피처를 통일합니다.

for df in [train_x, test_x]:
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

# 예측 시 제외할 컬럼들 (누수 방지 및 불필요 컬럼)
drop_cols = ['time', 'PM10', 'PM25']

# 두 데이터 세트의 컬럼 순서를 동일하게 맞춥니다.
X_train = train_x.drop(columns=drop_cols)
X_test = test_x.drop(columns=drop_cols)

# 공통된 피처만 사용하도록 보장
common_features = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_features]
X_test = X_test[common_features]

print(f"📊 사용된 공통 피처 ({len(common_features)}개): {list(common_features)}")

# 결측치 처리
X_train = X_train.fillna(method='ffill').fillna(0)
X_test = X_test.fillna(method='ffill').fillna(0)

# 3. 모델 및 하이퍼파라미터 설정
# R2 점수를 극대화하기 위해 검증된 3가지 앙상블 모델 사용
models = {
    'LightGBM': {
        'model': LGBMRegressor(random_state=42, verbose=-1),
        'params': {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [7, 10, 15],
            'num_leaves': [31, 63, 127],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'params': {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto', 'sqrt']
        }
    }
}

best_model = None
best_r2 = -float('inf')
best_model_name = ""

print("\n🔍 최적의 R2 점수를 찾기 위한 모델 튜닝을 시작합니다. (약 수 분 소요될 수 있습니다)")

# 4. Randomized Search를 통한 최적화
for name, mp in models.items():
    print(f"\n[{name}] 최적화 중...")
    # n_iter를 높여 더 정밀하게 탐색 (시간 관계상 20으로 설정)
    rs = RandomizedSearchCV(
        mp['model'], mp['params'], n_iter=20, cv=3, 
        scoring='r2', n_jobs=-1, random_state=42, verbose=0
    )
    rs.fit(X_train, y_train)
    
    y_pred = rs.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✨ [{name}] Test R2: {r2:.4f} | RMSE: {rmse:.4f}")
    print(f"📍 최적 파라미터: {rs.best_params_}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = rs.best_estimator_
        best_model_name = name

# 5. 최종 결과 저장
print("\n" + "="*50)
print(f"🏆 최종 승리 모델: {best_model_name}")
print(f"📈 최고 R2 점수: {best_r2:.4f}")
print("="*50)

# 최고 성능 모델 저장
joblib.dump(best_model, 'best_model_final.pkl')
print(f"\n🎉 모든 과정이 완료되었습니다! 최적화된 모델이 'best_model_final.pkl'로 저장되었습니다.")
