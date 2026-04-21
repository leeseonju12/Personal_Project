# Race Ready / F1 Analysis System

<img width="1536" height="1024" alt="LOGO_RaceReady_Light" src="https://github.com/user-attachments/assets/03d110ac-5043-4391-8af8-423e2312d8d6" />


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

## IntelliJ에서 Spring 실행

1. IntelliJ IDEA 실행
2. `Open` 선택
3. 아래 폴더 열기

```text
C:\Users\LEE\Project_lsj_work\Personal_Project\Race_Ready
```

4. Maven 로딩이 끝날 때까지 대기
5. `src/main/java/org/example/race_ready/RaceReadyApplication.java` 열기
6. `main()` 옆 초록 실행 버튼 클릭
7. `Run 'RaceReadyApplication'` 선택

정상 실행되면 로그에 아래와 비슷한 문구가 나옵니다.

```text
Tomcat started on port 8080
Started RaceReadyApplication
```

브라우저에서 접속합니다.

```text
http://localhost:8080
```

또는 직접 대시보드 API를 확인합니다.

```text
http://localhost:8080/api/dashboard
```

## Spring에서 Python을 직접 실행하는 방식

가장 빠르게 Python 결과를 웹에 그대로 보여주려면 Spring Controller에서 Python 파일을 실행하고 stdout을 HTML에 출력하면 됩니다.

실행해야 하는 Python 조합은 아래와 같습니다.

```text
실행 프로그램:
C:/Users/LEE/Project_lsj_work/Personal_Project/AI_Analysis_System/.venv/Scripts/python.exe

실행할 파일:
AI_Analysis_System/f1result.py

작업 디렉터리:
C:/Users/LEE/Project_lsj_work/Personal_Project
```

Java `ProcessBuilder` 예시는 아래와 같습니다.

```java
ProcessBuilder pb = new ProcessBuilder(
        "C:/Users/LEE/Project_lsj_work/Personal_Project/AI_Analysis_System/.venv/Scripts/python.exe",
        "AI_Analysis_System/f1result.py"
);

pb.directory(new File("C:/Users/LEE/Project_lsj_work/Personal_Project"));
pb.redirectErrorStream(true);
```

이 방식은 기존 Python 분석 코드를 거의 수정하지 않고 Spring 웹페이지에 결과를 표시할 수 있다는 장점이 있습니다.

## 실행 전 확인 사항

Python 가상환경이 정상인지 확인합니다.

```powershell
C:\Users\LEE\Project_lsj_work\Personal_Project\AI_Analysis_System\.venv\Scripts\python.exe -V
```

필요 패키지가 설치되어 있는지 확인합니다.

```powershell
C:\Users\LEE\Project_lsj_work\Personal_Project\AI_Analysis_System\.venv\Scripts\python.exe -c "import pandas, sqlalchemy, pymysql, requests, numpy; print('deps ok')"
```

Python 분석 코드가 단독으로 실행되는지 확인합니다.

```powershell
cd C:\Users\LEE\Project_lsj_work\Personal_Project
.\AI_Analysis_System\.venv\Scripts\python.exe AI_Analysis_System\f1result.py
```

Spring 웹 프로젝트가 실행되는지 확인합니다.

```powershell
cd C:\Users\LEE\Project_lsj_work\Personal_Project\Race_Ready
.\mvnw.cmd spring-boot:run
```

## 일반적인 문제 해결

`ModuleNotFoundError`가 발생하면 시스템 Python이 아니라 프로젝트 가상환경 Python을 사용하고 있는지 확인합니다.

```text
올바른 Python:
AI_Analysis_System/.venv/Scripts/python.exe
```

DB 연결 오류가 발생하면 MySQL 서버가 실행 중인지, `F1_DB_URL` 또는 `db.py`의 기본 URL이 맞는지 확인합니다.

Spring 웹에서 데이터가 안 보이면 `application.yaml`의 SQLite DB 경로가 실제 파일 위치와 일치하는지 확인합니다.

```yaml
race-ready:
  db-path: C:/Users/LEE/Project_lsj_work/Personal_Project/f1_analysis.db
```

`--starting-grid` 실행 결과가 비어 있으면 `meeting_key`, `session_key` 조합이 맞는지 확인해야 합니다.

## 권장 작업 흐름

1. `f1result.py`의 `TARGET_YEAR`, `TARGET_COUNTRY`를 원하는 경기로 변경
2. Python 단독 실행으로 분석 결과 확인
3. DB에 저장된 결과 또는 콘솔 출력 결과 확인
4. Spring Boot `Race_Ready` 실행
5. 브라우저에서 `http://localhost:8080` 또는 `/api/dashboard` 확인
