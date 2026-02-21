#!/usr/bin/env python3
"""Run the daily pipeline once for testing/debugging."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.pipeline.orchestrator import run_daily_pipeline

if __name__ == '__main__':
    app = create_app()
    target = date.today()

    if len(sys.argv) > 1:
        from datetime import datetime
        target = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()

    print(f"Running pipeline for {target}...")
    with app.app_context():
        brief = run_daily_pipeline(target)
        print(f"Pipeline complete. Brief status: {brief.status}")
        if brief.total_cost_usd:
            print(f"Total cost: ${brief.total_cost_usd:.4f}")
        if brief.idiot_index:
            print(f"Idiot index: {brief.idiot_index:.6f}")
