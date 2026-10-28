# ML WBS · Jira · GitHub 동기화

SCRUM-38 범위의 Python 3.12 표준 라이브러리 도구다. 학습 모듈을 가져오거나 원본 데이터를 읽지 않는다.

| WBS ID | Jira | 선행 ML 작업 |
| --- | --- | --- |
| 1.4 | SCRUM-33 | 없음 |
| 2.6 | SCRUM-34 | 없음 |
| 2.13 | SCRUM-35 | SCRUM-34 |
| 2.21 | SCRUM-36 | SCRUM-35 |
| 3.6 | SCRUM-37 | SCRUM-35, SCRUM-36 |

대상은 `dajaba-team/dajaba-ml`, Jira `SCRUM`, 설정에 고정된 염준영 계정과 WBS `준영` 단독 담당 5개 항목이다. 공동 담당·타인·소유 불명 항목은 쓰기 전에 전체 실행을 중단한다. 선행 관계는 관리 메모로 안내하며 다른 사람의 이슈나 Jira 링크를 자동 생성하지 않는다.

## 방향과 변경 필드

- WBS F열 종료일 → Jira `duedate`. WBS가 일정 기준이며 Jira 단독 일정 수정은 다음 실행에서 WBS 날짜로 맞춰진다. 일정 변경은 두 곳을 함께 반영한다.
- Jira 상태명 + 해당 Jira 키를 제목/본문에 정확히 포함한 GitHub PR → WBS H열의 자동 관리 **메모**. PR 정보는 읽기 전용이다.
- WBS 셀 값·완료율·시작일·수식·서식·사람 메모와 Jira 상태·담당자·설명은 자동 변경하지 않는다.
- 문서 작성, 코드 구현, 검증, 학습, 서비스 연동을 구분한다. 병합만으로 업무를 완료 처리하지 않는다. 완료 판단과 상태 전환은 근거를 확인한 담당자가 수행한다.

2026-10-28 사용자 지시로 SCRUM-33과 WBS 1.4 마감을 10/23에서 **10/30**으로 연장했다. 다른 일정은 유지했다.

## 실행

저장소 루트에서 실행한다. 아래 인증 값은 실행 환경에만 제공하고 파일·PR·로그에 기록하지 않는다.

- `JIRA_EMAIL`, `JIRA_API_TOKEN`: 대상 Jira를 읽고 담당 이슈 마감일을 수정할 수 있는 계정
- `GOOGLE_ACCESS_TOKEN`: 대상 스프레드시트를 읽고 수정할 수 있는 유효한 OAuth access token
- `GITHUB_TOKEN`: dajaba-ml 저장소 PR 읽기 권한

```powershell
python -m unittest discover -s tools/ml_sync -p 'test_*.py' -v
python tools/ml_sync/sync.py plan --plan sync-plan.json
python tools/ml_sync/sync.py apply --plan sync-plan.json --journal sync-journal.json
```

`plan`은 읽기만 한다. `apply`는 현재 전체 스냅샷으로 계획을 다시 계산하고, 각 쓰기 직전 담당자·매핑·변경 전 값을 재확인하고 쓰기 후 재조회한다. 계획 변조·오래된 스냅샷·중복 ID·담당자 변경·손상된 관리 구간·수식 마감일은 중단 사유다. 재실행 시 동일 상태는 변경 0건이다. API의 읽기와 쓰기 사이에는 원자적 비교 갱신이 없어 극히 짧은 동시 편집 구간은 남는다. 실행 중 대상 편집을 피하고 충돌 시 새 계획을 검토한다.

자동 관리 메모는 BEGIN/END 구간만 교체한다. 고정 WBS ID와 Jira 라벨을 사용하므로 행 이동에도 대응한다. 이슈나 PR을 생성하는 기능이 없어 반복 실행으로 중복 생성하지 않는다. 부분 성공은 journal의 `verified`와 `write_returned`로 구분한다. 실패 시 자동 롤백·맹목 재시도를 하지 않고 현황 재조회 후 새 계획을 만든다. 잠금 파일은 동일 journal 경로의 로컬 중복 실행을 막으며 GitHub Actions는 concurrency로 직렬화한다. 로컬과 Actions의 동시 실행은 피한다.

오프라인 MCP 스냅샷은 `plan --snapshot <파일>`에만 사용할 수 있고 `apply` 입력으로 허용되지 않는다. MCP를 통한 실제 반영은 동일 규칙으로 계획을 검토하고, 쓰기 직전 재조회 및 반영 후 비교를 별도로 수행한다.

## GitHub 실행과 운영 미결

`ml-sync-check.yml`은 develop 대상 PR의 브랜치/Jira 링크/본문 형식과 단위 테스트를 검사한다. Gitmoji + Type 명사형 커밋을 사용하고 동료 최소 1명의 승인 후 develop에 병합한다. 브랜치 보호나 프로젝트 전체 설정은 변경하지 않는다.

`ml-sync-manual.yml`은 develop에서 **수동 실행**하며 기본값은 계획만 생성한다. `apply=true`일 때만 실제 쓰기를 수행한다. 사용 전 저장소의 `ML_SYNC_JIRA_EMAIL`, `ML_SYNC_JIRA_API_TOKEN`, `ML_SYNC_GOOGLE_ACCESS_TOKEN` secrets가 필요하다. GitHub 토큰은 해당 실행의 읽기 전용 토큰을 사용한다. 플러그인 연결의 인증은 GitHub Actions로 전달되지 않는다. 이번 작업에서 secrets를 등록하거나 원격 인증 실행을 검증하지 않았다. Google access token은 만료되므로 수동 실행 전 갱신이 필요하며 장기 예약 운영에는 별도 인증 갱신 구성이 필요하다.

반복 일정은 아직 미정이다. 주간 동기화 요구는 있으나 요일·시각·시간대가 확정되지 않아 cron과 예약 작업을 만들지 않았다. PR 병합·승인과 인증 설정 후 실행할 수 있다. 프로젝트 전체의 Jira GitHub 앱 설치 설정을 바꾸지 않고, 이슈 키·링크와 API 기반 조회로 연동한다. Jira 개발 패널의 브랜치/PR 자동 노출 여부는 별도 미확인이다.
