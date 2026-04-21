# Race Ready / F1 Analysis System

<img width="768" height="512" alt="LOGO_RaceReady_Light" src="https://github.com/user-attachments/assets/296e0e80-9586-4939-96c1-9f0cd98bffc1" />


F1 데이터를 수집하고 분석한 뒤, 결과를 콘솔 또는 Spring Boot 웹페이지에서 확인하는 프로젝트입니다.

현재 프로젝트는 크게 두 부분으로 나뉩니다.

- `AI_Analysis_System`: Python 기반 F1 데이터 수집/분석 코드
- `Race_Ready`: Spring Boot 기반 웹 프로젝트

## 프로젝트 구조

```text
Personal_Project/
  README.md
  f1_analysis.db

  AI_Analysis_System/
    f1result.py
    DB.sql
    DBtest.py
    f1_analysis.db
    .venv/
    f1_analytics/
      db.py
      models_finish_group.py
      models_pace_band.py
      models_quali_analysis.py
      models_strategy_analysis.py
      backtest_evaluator.py
      pipelines/
        run_meeting.py
        run_season.py
        pipeline_season.py

  Race_Ready/
    pom.xml
    mvnw
    mvnw.cmd
    src/main/java/org/example/race_ready/
      RaceReadyApplication.java
      f1web/F1ResultController.java
    src/main/resources/
      application.yaml
      templates/result.html
      static/index.html
```

## Python 분석 코드

Python 실행 진입점은 아래 파일입니다.

```text
AI_Analysis_System/f1result.py
```

기본 실행 시 `TARGET_YEAR`, `TARGET_COUNTRY` 값으로 조회할 대상을 결정합니다.

```python
TARGET_YEAR = int(os.getenv("F1_YEAR", "2025"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Mexico")
```

예를 들어 2024년 일본 GP를 조회하려면 아래처럼 바꾸면 됩니다.

```python
TARGET_YEAR = int(os.getenv("F1_YEAR", "2024"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Japan")
```

기본 분석 대상만 바꾸는 경우에는 이 두 값만 수정하면 됩니다. 다른 분석 로직은 건드릴 필요가 없습니다.

## Python 실행 방법

PowerShell에서 프로젝트 루트로 이동합니다.

```powershell
cd C:\Users\LEE\Project_lsj_work\Personal_Project
```

가상환경 Python으로 `f1result.py`를 실행합니다.

```powershell
.\AI_Analysis_System\.venv\Scripts\python.exe AI_Analysis_System\f1result.py
```

이 명령은 `TARGET_YEAR`, `TARGET_COUNTRY` 기준으로 데이터를 조회하고 분석 결과를 콘솔에 출력합니다.

## 환경변수로 조회 대상 바꾸기

코드를 직접 수정하지 않고 실행할 때만 조회 대상을 바꿀 수도 있습니다.

```powershell
$env:F1_YEAR="2024"
$env:F1_COUNTRY="Japan"
.\AI_Analysis_System\.venv\Scripts\python.exe AI_Analysis_System\f1result.py
```

환경변수를 사용하면 `f1result.py`의 기본값보다 환경변수 값이 우선 적용됩니다.

우선순위는 아래와 같습니다.

```text
F1_YEAR 환경변수 값이 있으면 사용
없으면 f1result.py의 기본값 2025 사용

F1_COUNTRY 환경변수 값이 있으면 사용
없으면 f1result.py의 기본값 Mexico 사용
```

## Starting Grid 조회

`--starting-grid` 옵션을 사용하면 `TARGET_YEAR`, `TARGET_COUNTRY`가 아니라 `meeting_key`, `session_key`를 기준으로 시작 그리드를 조회합니다.

```powershell
.\AI_Analysis_System\.venv\Scripts\python.exe AI_Analysis_System\f1result.py --starting-grid --meeting-key 1264 --session-key 9951
```

정리하면 실행 모드별 조회 기준은 아래와 같습니다.

```text
python f1result.py
-> TARGET_YEAR, TARGET_COUNTRY 사용

python f1result.py --starting-grid --meeting-key ... --session-key ...
-> meeting_key, session_key 사용
```

## DB 설정

Python DB 연결 설정은 아래 파일에 있습니다.

```text
AI_Analysis_System/f1_analytics/db.py
```

현재 기본 DB URL은 MySQL입니다.

```text
mysql+pymysql://root@127.0.0.1:3306/Spring_project_26_02?charset=utf8mb4
```

다른 DB를 사용하려면 `F1_DB_URL` 환경변수로 바꿀 수 있습니다.

```powershell
$env:F1_DB_URL="mysql+pymysql://root@127.0.0.1:3306/Spring_project_26_02?charset=utf8mb4"
.\AI_Analysis_System\.venv\Scripts\python.exe AI_Analysis_System\f1result.py
```

DB 스키마는 아래 파일에 정의되어 있습니다.

```text
AI_Analysis_System/DB.sql
```

## Spring Boot 웹 프로젝트

Spring Boot 웹 프로젝트는 아래 폴더입니다.

```text
Race_Ready
```

주요 파일은 아래와 같습니다.

```text
Race_Ready/pom.xml
Race_Ready/src/main/java/org/example/race_ready/RaceReadyApplication.java
Race_Ready/src/main/java/org/example/race_ready/f1web/F1ResultController.java
Race_Ready/src/main/resources/application.yaml
Race_Ready/src/main/resources/templates/result.html
Race_Ready/src/main/resources/static/index.html
```

웹 프로젝트는 현재 SQLite DB 파일을 읽어서 대시보드 API를 제공합니다.

```yaml
race-ready:
  db-path: C:/Users/LEE/Project_lsj_work/Personal_Project/f1_analysis.db
```

DB 파일 위치가 바뀌면 `Race_Ready/src/main/resources/application.yaml`의 `race-ready.db-path` 값을 수정해야 합니다.

## 권장 작업 흐름

1. `f1result.py`의 `TARGET_YEAR`, `TARGET_COUNTRY`를 원하는 경기로 변경
2. Python 단독 실행으로 분석 결과 확인
3. DB에 저장된 결과 또는 콘솔 출력 결과 확인
4. Spring Boot `Race_Ready` 실행
5. 브라우저에서 `http://localhost:8080` 또는 `/api/dashboard` 확인
