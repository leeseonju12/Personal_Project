import os

from f1_analytics.db import get_engine
from f1_analytics.pipelines.run_meeting import run_meeting_by_country

TARGET_YEAR = int(os.getenv("F1_YEAR", "2025"))
TARGET_COUNTRY = os.getenv("F1_COUNTRY", "Mexico")

def main() -> None:
    engine = get_engine()
    outputs = run_meeting_by_country(engine, year=TARGET_YEAR, country_name=TARGET_COUNTRY)
    print(f"[DONE] run finished: country={TARGET_COUNTRY}, sessions={len(outputs)}")
    if outputs:
        forecast = outputs[0]["forecast"].sort_values("exp_points", ascending=False)
        print(forecast.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

