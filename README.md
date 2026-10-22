# 대기·기상 데이터 기반 미세먼지(PM10) 농도 예측

AI+X 중급 프로젝트(2조, 팀 프로젝트)입니다. 대기질·기상 데이터(2021–2022)를 정제하고 여러 회귀 모델을 비교해 PM10 농도를 예측했습니다.

## 본인 역할
- 데이터 EDA (분포·계절 패턴·상관 분석)
- 머신러닝 모델링: 선형회귀·SVR·Random Forest·XGBoost·Gradient Boosting 비교, 그리드 서치 튜닝, PCA·특징 추가 실험
- 딥러닝(ANN)과 시계열 모델링은 다른 팀원이 담당했습니다.

## 결과
| 단계 | R² |
|---|---|
| 베이스라인 최고 모델 (SVR) | 0.642 (RMSE 14.17) |
| 그리드 서치 튜닝 | 0.681 |
| PCA 적용 | 0.686 |
| 전일 베이징 풍향·풍속 기반 해외 유입 특징 추가 | 0.689 |

- 다중회귀로는 '매우 나쁨' 구간을 예측하지 못했고, 고농도 이상치 예측이 한계로 남았습니다.

## 구성
- `1_Exploratory_Data_Analysis.ipynb` 분포·계절 패턴 탐색
- `2_Data_Preprocessing.ipynb` 결측치 처리, 이상치 필터링, VIF 다중공선성 검정
- `3_Machine_Learning_Modeling.ipynb` 모델 비교와 그리드 서치
- `vif_correlation_check.py`, `svr_importance.py`, `gb_tuning.py`, `linear_regression.py`, `best_model.py` 개별 실험 코드
- `AI_X_Finedust_Combined_Report.pdf` 팀 보고서, `AI_X_Finedust_Individual_Report.pdf` 개인 보고서
- `*.png` 결과 그림

## 데이터 출처
- 대기질: 에어코리아(한국환경공단) 측정 자료
- 기상: 기상청 기상자료개방포털
- 수업에서 제공받은 가공본(`air_*.csv`, `weather_*.csv`)과 전처리 결과(`train_*`, `test_*`)를 함께 두었습니다.

## 기술
Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter
