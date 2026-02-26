from sqlalchemy import create_engine, text

DB_URL = "mysql+pymysql://root@127.0.0.1:3306/Spring_project_26_02?charset=utf8mb4"

try:
    engine = create_engine(DB_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT DATABASE();"))
        print("✅ 연결 성공")
        print("현재 DB:", result.fetchone())

except Exception as e:
    print("❌ 연결 실패")
    print("에러:", e)