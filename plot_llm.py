#!/usr/bin/env python3
"""
Convenience wrapper: runs the LLM replica sweep visualization.

Usage:
    cd serve-performance-blog
    python plot_llm.py [results_dir]
"""

import subprocess
import sys

args = ["python", "llm/scripts/visualize_replica_sweep.py"]
if len(sys.argv) > 1:
    args.append(sys.argv[1])
else:
    args.append("llm/results/replica_sweep")

subprocess.run(args, check=True)
