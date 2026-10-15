"""
SentinX - run the full data pipeline: ACLED -> World Bank -> merge.

Usage (from repo root):
    python pipeline/run_pipeline.py [path/to/acled.xlsx]
"""
import sys

import acled_country_year
import merge
import worldbank


def main() -> None:
    acled_path = sys.argv[1] if len(sys.argv) > 1 else acled_country_year.DEFAULT_RAW
    print("=== 1/3 ACLED ===")
    acled_country_year.main(acled_path)
    print("\n=== 2/3 World Bank ===")
    worldbank.main()
    print("\n=== 3/3 Merge ===")
    merge.main()


if __name__ == "__main__":
    main()
