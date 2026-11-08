"""Run every stage in dependency order from the repository root:  python run_project.py

generate_data.py is executed unchanged as a subprocess; nothing here needs a network or an API key.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print("=== RAW DATA: generate_data.py ===")
    subprocess.run([sys.executable, "generate_data.py"], cwd=ROOT, check=True)

    from src import data_pipeline, sql_analysis, customer_models, triage, policy_rag, run_scenarios

    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    data_pipeline.audit()
    data_pipeline.clean_data()
    ok = data_pipeline.validate()
    print("\n=== PART A: SQL + PANDAS ANALYSIS ===")
    ok &= bool(sql_analysis.main())
    print("\n=== PART B: SEGMENTS + LAPSE MODEL ===")
    customer_models.main()
    print("\n=== PART C: TRIAGE ===")
    triage.main()
    print("\n=== PART D: POLICY RAG ===")
    policy_rag.main()
    print("\n=== PART D: REFUND AGENT SCENARIOS ===")
    ok &= bool(run_scenarios.main())
    print("\nAll stages finished." if ok else "\nFinished WITH FAILURES; see output above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
