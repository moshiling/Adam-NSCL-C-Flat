#!/usr/bin/env bash
set -euo pipefail
cd /home/moshiling
/usr/bin/env python3 /home/moshiling/Adam-NSCL-C-Flat/outputs/selective_g0_smalllambda_round/pipeline.py stage1-group --group B --gpu "${1:-5}"
