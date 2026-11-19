# SCRUM-130 연결 시험용 릴리스

`aml-ml 0.1.0rc1`은 Python 3.12용입니다. `aml_ml`에서 `AlertInput`, `Coverage`, `score`, `MODEL_VERSION`, `FEATURE_VERSION`, `THRESHOLD`를 가져올 수 있습니다. 모델은 기존 SCRUM-130 XGBoost 후보를 사용합니다. 이후에 학습한 HI 전용 모델과 DL의 배포 여부는 별도입니다.

| 입력 | 패키지 결과 |
| --- | --- |
| R01·R02·R04와 유효한 입력 | 점수, `OK`, 설명 최대 3개 |
| R03·R05·R06 | `score=None`, `FAILED`, `RULE_NOT_SUPPORTED`, `reasons=[]` |
| 지원 룰의 24시간 이력 미확인 또는 부족 | `score=None`, `FAILED`, `HISTORY_NOT_SUPPORTED` |

`RULE_NOT_SUPPORTED`는 패키지 결과 코드입니다. AI의 현재 `/v1/scores` 스키마에는 이 값이 없으므로, 실제 AI 연동에서 이 값을 다룰 방식을 확정해야 합니다. 지원되지 않는 룰 실패를 이력 부족이나 반복 재시도 대상으로 분류하면 의미가 달라집니다. `THRESHOLD=0.5`는 패키지 추천값이며 운영 경계는 백엔드 정책을 따릅니다.

백엔드 v3 샘플에서는 지원 룰 25건에 점수를 냈고 미지원 룰 17건은 별도 상태로 반환했습니다. 묶음 반복 결과와 R04 개별 계산이 일치했고, R05 대표 경로 밖 근거가 포함된 항목도 묶음 처리를 중단시키지 않았습니다. 저장소의 고정 테스트는 새로 만든 합성 계좌·거래 및 이를 변형한 혼합 묶음을 사용합니다.
