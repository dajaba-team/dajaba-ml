# 사건형 R04 채점 후보 재현

공식 `v0.1.0rc2`는 기존 HI+LI Medium 모델이다. 이 브랜치의 `0.1.0rc3`은 사건형 R04를 반영해 HI-Medium으로 재학습한 최종 후보의 소스 버전이다. AI 저장소의 공식 rc2 의존성, 서버 배포와 시연 정책은 기존 상태에 따른다.

| 항목 | 후보 계약 |
|---|---|
| 모델 버전 | `xgb-hi-medium-r04-event-c2-31-v1` |
| 피처 버전 | `hi-medium-evidence31-96h-r04-event-v1` |
| 내부 체크포인트 | `r04_original.ubj` |
| SHA-256 | `985be923c53fcbd9b5912e81f99911b598c38340d1ad7a65d59bc2a74fd076e2` |
| 특징 순서 | [serving_contract.json](../src/aml_ml/serving_contract.json)의 31개 `float32` 피처 |

Python 3.12 가상환경에서 실행한다.

~~~sh
python -m pip install . pytest
python -m pip check
python -m pytest tests/test_rc3_serving_contract.py -q
~~~

합성 거래와 고정 점수 대역으로 피처 순서, 정상 결과, 체크포인트 가드를 검사한다. 이 공개 검사는 실제 모델 추론과 구분한다.

실제 채점에는 배포 담당자가 제공한 승인된 체크포인트를 사용한다. 셸 환경 변수 `AML_ML_MODEL_PATH`에 해당 로컬 파일 경로를 지정한다. 예를 들어 Bash에서 `export AML_ML_MODEL_PATH=./private-models/r04_original.ubj`, PowerShell에서 `$env:AML_ML_MODEL_PATH = './private-models/r04_original.ubj'`로 지정한다. `python -c "from aml_ml import load_candidate; load_candidate()"`로 SHA-256과 피처 순서를 확인한다.

`aml_ml.score(alerts, transactions, coverage=coverage)`에 `AlertInput` 목록, 거래 DataFrame, `Coverage` 목록을 넘긴다. 거래 필드는 `serving_candidate.TX_COLUMNS`를 사용한다. 금액은 Decimal, 기준 시각은 분 단위, 전체 경보 근거와 확인된 직전 24시간 커버리지를 공급한다. `features(..., coverage=coverage)`는 모델 파일 없이 계산 결과를 검사할 수 있다. 서비스의 거래 조회와 예측 저장은 AI 저장소에서 담당한다.
