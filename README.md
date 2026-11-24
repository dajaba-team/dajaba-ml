# dajaba-ml

**MLDB 조립 데이터로 XGBoost를 학습·평가하고, AI 서버가 사용하는 경보별 공식 ML 점수를 계산합니다.**

HI-Medium의 C2 proxy 레이블로 학습한 사건형 R04 모델의 채점 코드를 `aml-ml` 패키지로 제공합니다. `v0.1.0rc3`의 채점 룰은 R01~R06입니다. 아래 2026-11-08 전처리 기록은 당시 오프라인 룰을 기준으로 합니다.

## 시스템 아키텍처

![MLDB 조립 데이터로 XGBoost를 학습하고 AI 서버가 rc3 패키지로 경보를 채점하는 흐름](docs/images/ml-architecture-ci.png)

AI 서버의 `POST /v1/scores`는 실제 채점 모드에서 `v0.1.0rc3`의 `aml_ml.score()`를 호출합니다. 패키지는 96시간 경보 창의 근거와 직전 24시간 이력의 완전성을 확인한 뒤, 학습 때와 같은 순서의 `float32` 피처 31개로 경보당 0~1 점수를 계산합니다. 위험 등급 판정과 케이스 생성은 백엔드 정책이 담당합니다. 가중치는 승인된 내부 체크포인트에서 읽습니다. [공식 태그](https://github.com/dajaba-team/dajaba-ml/tree/v0.1.0rc3) · [설치·채점 재현](docs/rc3-serving.md)

## 초기 오프라인 전처리 범위 (2026-11-08)

ml_preparation 패키지를 한 단위로 관리합니다. 같은 패키지 안에서 전처리와 학습이 공유하는 기능을 폴더 이름만으로 나누지 않습니다.

- 당시 오프라인 전처리 범위: R01·R02·R04·R06·R07.
- HI·LI: 다섯 룰. SAML: R01·R02·R04·R06, R07은 적용 불가로 구분.
- 2026-11-08 실행의 산출물 생성과 포맷 검증은 완료 보고됨. 모델 학습·성능 평가·서비스 ID 교차검증 완료를 뜻하지 않습니다.
- 기존 학습 코드를 다중 룰 산출물에 연결하는 어댑터와 평가 수용 기준은 별도 검증 대상입니다.
- 최신 리팩터링은 정적 검토와 사용자 실행 회귀 검증을 구분해서 인수합니다.

## 초기 오프라인 적용 룰

| 룰 | 초기 적용 조건 |
| --- | --- |
| R01 | 24h 서로 다른 수신 계좌 5 이상 |
| R02 | 24h 서로 다른 송신 계좌 5 이상 |
| R04 | 72h, 최대 3홉 순환, 금액 조건 없음 |
| R06 | GATHER_SCATTER: 24h 입금 상대 3 이상 AND 출금 상대 3 이상, 선후 무관 |
| R07 | US Dollar 입금 [8,000, 10,000), 3건 이상, Cash 한정 없음 |

R01·R02·R07은 B7 분 단위 24h 창과 거짓→참 경보 생성 기준을 따릅니다. R06의 기존 BIPARTITE 의미를 재사용하지 않습니다. R03·R05·R08은 이번 실행 범위 밖입니다.

초기 오프라인 룰의 조건값은 [공개 설정 예제](configs/offline_rules.example.json)의 `policy.rules`에서 확인할 수 있습니다. 오프라인 rule_versions는 백엔드 배포 버전과 별도 네임스페이스입니다. 백엔드 v1과 문자열이 같다고 가정하지 않고, 조건·창·근거를 대조합니다.

## SAML의 한계

SAML의 단일 Amount를 지급액과 수취액 양쪽으로 복제하지 않습니다. 방향별 원통화 금액은 확인되지 않은 값을 null로 유지합니다. US Dollar 수취액을 요구하는 R07은 적용 불가이며 정상 0건으로 표시하지 않습니다. 금액 분포가 비슷하다는 이유만으로 HI·LI와 의미가 같다고 추정하지 않습니다.

## 코드 안내

| 위치 | 역할 |
| --- | --- |
| ml_preparation/src/aml_ml/offline_multi.py | 전체 실행 순서 |
| offline_rule_sql.py | 룰별 후보 탐지 SQL |
| offline_multi_io.py | 입력 읽기와 출력 I/O |
| offline_multi_metadata.py | 경보·레이블·분할 메타데이터 |
| offline_multi_dataset.py | 피처와 X·y 출력 |
| offline_multi_recovery.py | 중단·재개 전 확인 |
| offline_multi_progress.py / offline_multi_diagnostics.py | 진행 기록과 진단 |
| preprocessing.py / offline_training.py / offline_xgboost.py | 전처리와 기존 학습 구현 |
| ml_preparation/tests | 합성 데이터와 회귀 검사 |
| ml_preparation/configs | 룰·입력·실행 설정 |

패키지 소스는 이 저장소의 `src/aml_ml`에 있습니다. 표의 ml_preparation은 기존 로컬 프로젝트에서 사용한 경로명입니다.

## 데이터 계약

- 피처는 reference_ts보다 엄격히 이전의 관측만 사용합니다.
- 전체 경보 근거와 대상 계좌의 직접 입출금 집계는 구분합니다.
- 경로 중간 거래는 전체 근거로 보존하되 대상 계좌의 직접 금액에 더하지 않습니다.
- 원본 정답·pattern·ID는 모델 입력 피처가 아닙니다. proxy 레이블은 조사 확정 결과가 아닙니다.
- train에서만 전처리를 적합하고 validation으로 선택합니다. test는 선택 후 평가합니다.
- 분할 간 근거 중복 방지는 계좌·그래프 전체 독립성 보장이 아닙니다.
- 학습 UID를 실제 서비스 transaction_id로 추정하지 않습니다. 서비스 매핑은 원본 SHA-256과 행 번호를 기준으로 검증합니다.

## 환경과 사용

Python 3.12, DuckDB·PyArrow·Polars·Pandas·NumPy·XGBoost를 사용합니다. 설치 기준은 루트 `pyproject.toml`입니다. [rc3 채점 패키지 설치와 검증](docs/rc3-serving.md)을 참고합니다.

오프라인 입력 인덱스와 일부 통합 테스트는 MLDB 작업 공간에 의존합니다. rc3 채점 계약의 합성 회귀 검사는 이 저장소에서 실행할 수 있습니다. 전체 전처리 런처는 검사가 끝나면 데이터 생성까지 진행하므로 테스트만 할 때 사용하지 않습니다.

## CI/CD 제안

PR에서 정적 검사와 합성 데이터 회귀를 실행하는 것은 ML 저장소 CI의 제안 사항입니다. PR에서 원천 전체 처리나 모델 학습, 운영 DB 접근을 자동 실행하지 않습니다.

## 함께 보기

[다자바 MLDB](https://github.com/dajaba-team/dajaba-mldb)는 원천 검사와 기초 데이터 조립을 맡습니다. 공개 저장소에는 코드·안전한 설정 예시·합성 테스트·설명 문서만 올립니다.

## 현재 검증 진입점

rc3 채점 패키지의 공개 회귀는 `python -m pytest tests/test_rc3_serving_contract.py -q`로 실행합니다. 기존 MLDB 통합 검사는 두 데이터 패키지를 함께 설치한 뒤 아래에 적힌 방법으로 실행합니다.

현재 src/aml_ml은 MLDB 패키지를 import하지 않지만 tests/test_prepare.py는 DBcheck·AssembleDB로 합성 통합 입력을 생성합니다. 실제 폴더의 전체 회귀는 세 패키지를 함께 연결해 검사합니다. 별도 ML 저장소에서는 자체 검사와 이 통합 검사를 구별하고, MLDB 버전을 명시해야 합니다.

## 별도 ML 저장소에서의 설치와 검사

이 폴더 자체가 ML 저장소 루트가 됩니다. 아래 명령은 사용자 실행용이며 이번 변경에서는 실행하지 않았습니다.

```powershell
python -m pip install -e '.[training]' 'pytest>=8,<10' 'ruff>=0.9,<1'
python -m pytest tests --ignore=tests/test_prepare.py -q -p no:cacheprovider
```

test_prepare.py는 MLDB의 DBcheck와 AssembleDB를 사용하는 합성 통합 검사입니다. 삭제하거나 전체 통과에서 누락시키지 않고 별도로 두 패키지를 검토한 로컬 경로에서 설치한 후 실행합니다. ML 런타임 자체는 이 두 패키지를 import하지 않습니다.

```powershell
# 검토된 MLDB checkout 경로를 입력합니다. 동명 PyPI 패키지를 임의 설치하지 않습니다.
$mldb = Read-Host '검토된 MLDB 저장소 절대 경로'
python -m pip install -e (Join-Path $mldb 'DBcheck') -e (Join-Path $mldb 'AssembleDB')
python -m pytest tests/test_prepare.py -q -p no:cacheprovider
```

configs에는 공개 예제만 게시합니다. 기존 운영 설정은 로컬에 유지하며 tests/fixtures/legacy는 과거 N=10 및 미확정 정책의 회귀 검사 전용입니다. scripts/run_offline_rules_v1.py는 기존 고정 실행의 회귀 대상이며, 새 checkout의 범용 실행 명령으로 사용하지 않습니다.
