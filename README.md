# Race Ready

<p align="center">
  <a href="https://ibb.co/ym85TFhB">
    <img src="https://ibb.co/ym85TFhB" alt="Race Ready 메인 로고" width="260" />
  </a>
</p>

Race Ready는 F1 레이스 데이터를 기반으로 예선 성능, 예선 예측, 레이스 예측, 랩 페이스 밴드, 피트 전략 분석을 한 화면에서 확인할 수 있는 대시보드입니다. Spring Boot 백엔드가 SQLite DB를 조회해 `/api/dashboard`로 데이터를 제공하고, React 대시보드가 해당 API를 호출해 시각화합니다.

## 주요 기능

- 예선 성능 Top 10 표시
- 드라이버별 베스트랩, QPI, 예상 그리드 정보 제공
- Pole, Top 3, Top 10 확률 기반 예선 예측
- 기대 포인트, Top 10 확률, 포디움 확률 기반 레이스 예측
- RPI 기반 Lap Delta Band 시각화
- 피트스톱 횟수, 피트 타임, 손실 분위수 기반 전략 분석
- `result_f1.sql` 기준의 드라이버명, 팀명, 베스트랩, 그리드 데이터 보정

## 기술 스택

- Backend: Spring Boot, Spring Web MVC, Thymeleaf
- Frontend: React, TypeScript, Vite
- Database: SQLite
- Build: Maven, npm

## 프로젝트 구조

```text
Race_Ready/
├─ src/
│  ├─ frontend/
│  │  ├─ main.tsx              # React 앱 엔트리
│  │  └─ styles.css            # 전역 스타일
│  ├─ main/
│  │  ├─ java/
│  │  │  └─ org/example/race_ready/
│  │  │     ├─ RaceReadyApplication.java
│  │  │     └─ f1web/
│  │  │        └─ F1ResultController.java
│  │  └─ resources/
│  │     ├─ application.yaml   # DB 경로 및 Spring 설정
│  │     ├─ static/            # Vite 빌드 결과물
│  │     └─ templates/
│  │        ├─ result.html
│  │        └─ result.tsx      # 대시보드 React 컴포넌트
│  └─ test/
├─ index.html                  # Vite HTML 엔트리
├─ package.json                # React/Vite 의존성 및 npm 스크립트
├─ pom.xml                     # Spring Boot/Maven 설정
├─ result_f1.sql               # 대시보드 표시용 DB 보정 SQL
├─ tsconfig.json               # TypeScript 설정
└─ vite.config.ts              # Vite 빌드 설정
```

## 동작 구조

1. Spring Boot 서버가 SQLite DB를 조회합니다.
2. `/api/dashboard` 엔드포인트가 대시보드에 필요한 데이터를 JSON으로 반환합니다.
3. `result.tsx` React 컴포넌트가 `http://localhost:8080/api/dashboard`를 호출합니다.
4. Vite로 빌드된 React 앱이 Spring 정적 리소스(`src/main/resources/static`)로 제공됩니다.
5. 사용자는 `/f1` 또는 `/` 경로에서 React 대시보드를 확인합니다.

## 사전 준비

다음 도구가 설치되어 있어야 합니다.

- Java 17
- Node.js 및 npm
- Maven Wrapper 사용 가능 환경

SQLite DB 경로는 [application.yaml](src/main/resources/application.yaml)에 설정되어 있습니다.

```yaml
race-ready:
  db-path: C:/Users/LEE/Project_lsj_work/Personal_Project/f1_analysis.db
```

환경이 다르면 `race-ready.db-path` 값을 실제 DB 위치에 맞게 수정하세요.

## 설치 방법

프론트엔드 의존성을 설치합니다.

```powershell
npm install
```

Spring Boot 의존성은 Maven Wrapper가 자동으로 처리합니다.

```powershell
.\mvnw.cmd test
```

## 실행 방법

React 코드를 Spring Boot 정적 리소스로 빌드합니다.

```powershell
npm run build
```

Spring Boot 서버를 실행합니다.

```powershell
.\mvnw.cmd spring-boot:run
```

브라우저에서 다음 주소로 접속합니다.

```text
http://localhost:8080/f1
```

루트 경로도 `/f1`로 이동합니다.

```text
http://localhost:8080/
```

## 개발 모드

React 개발 서버만 따로 실행하려면 다음 명령을 사용합니다.

```powershell
npm run dev
```

Vite 개발 서버는 기본적으로 다음 주소에서 실행됩니다.

```text
http://localhost:5173
```

이때 Spring Boot API 서버도 별도로 실행되어 있어야 합니다.

```powershell
.\mvnw.cmd spring-boot:run
```

## API 확인

대시보드 데이터는 다음 API에서 JSON 형태로 확인할 수 있습니다.

```text
http://localhost:8080/api/dashboard
```

반환 데이터는 다음 섹션으로 구성됩니다.

- `qualifyingSummary`
- `qualifyingForecast`
- `raceForecast`
- `lapDeltaBand`
- `strategyAnalysis`

## DB 보정 로직

`result_f1.sql`은 화면에 필요한 드라이버명, 팀명, 베스트랩, 그리드 데이터를 보정하는 기준 SQL입니다. 현재 Spring API는 이 SQL의 핵심 내용을 SQLite 환경에 맞게 반영합니다.

`/api/dashboard` 호출 시 다음 작업이 수행됩니다.

- `dim_driver` 테이블 생성
- `driver_name`, `team_name` 메타데이터 추가 및 업데이트
- `feat_driver_session_metrics.best_lap_ms` 컬럼 보정
- `feat_driver_session_metrics.qpi_pct` 컬럼 보정
- `forecast_quali.starting_grid_position` 컬럼 보정
- 화면 섹션별 조회 쿼리에서 보정된 값 반환

## 주요 명령어

```powershell
# 타입 검사
npm run typecheck

# React 앱 빌드
npm run build

# Spring Boot 테스트
.\mvnw.cmd test

# Spring Boot 실행
.\mvnw.cmd spring-boot:run
```

## 화면 접속 경로

```text
React 대시보드: http://localhost:8080/f1
API 데이터:     http://localhost:8080/api/dashboard
```

`/api/dashboard`에 접속하면 JSON이 보이는 것이 정상입니다. React 화면은 `/f1`에서 확인해야 합니다.
