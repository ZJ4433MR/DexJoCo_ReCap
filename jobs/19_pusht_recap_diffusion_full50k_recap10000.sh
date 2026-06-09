#!/usr/bin/env bash
set -euo pipefail

# PuSH-T full-data diffusion ReCap step sweep.
# Same as jobs/17, but extends only the ReCap policy fine-tune to 10,000 steps.
export TAG="${TAG:-pusht_recap_diffusion_full50k_w15_r10000}"
export RECAP_POLICY_STEPS="${RECAP_POLICY_STEPS:-10000}"

exec bash jobs/17_pusht_recap_diffusion_full50k.sh
