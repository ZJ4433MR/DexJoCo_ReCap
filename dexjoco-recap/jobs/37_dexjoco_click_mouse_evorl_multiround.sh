#!/usr/bin/env bash
set -euo pipefail

cd "$EXP_DIR"
source scripts/dexjoco_common.sh

export DEXJOCO_TASK="${DEXJOCO_TASK:-click_mouse}"
export DEXJOCO_EVAL_SEED="${DEXJOCO_EVAL_SEED:-0}"
export DEXJOCO_EVAL_EPISODES="${DEXJOCO_EVAL_EPISODES:-100}"
export DEXJOCO_EVO_ROUNDS="${DEXJOCO_EVO_ROUNDS:-3}"
export DEXJOCO_EVO_MERGE_POOL="${DEXJOCO_EVO_MERGE_POOL:-1}"
export DEXJOCO_EVO_INITIAL_COLLECT_EPISODES="${DEXJOCO_EVO_INITIAL_COLLECT_EPISODES:-0}"
export DEXJOCO_EVO_INITIAL_COLLECT_SEED="${DEXJOCO_EVO_INITIAL_COLLECT_SEED:-$((DEXJOCO_EVAL_SEED + 10000))}"
export DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES="${DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES:-100}"
export DEXJOCO_EVO_INITIAL_POOL_INPUTS="${DEXJOCO_EVO_INITIAL_POOL_INPUTS:-}"
export DEXJOCO_EVO_COLLECT_EPISODES="${DEXJOCO_EVO_COLLECT_EPISODES:-120}"
export DEXJOCO_EVO_TRAIN_STEPS="${DEXJOCO_EVO_TRAIN_STEPS:-1200}"
export DEXJOCO_EVO_POSITIVE_REPEAT="${DEXJOCO_EVO_POSITIVE_REPEAT:-5}"
export DEXJOCO_EVO_PROMPT_AGENTS="${DEXJOCO_EVO_PROMPT_AGENTS:-0}"
export DEXJOCO_EVO_PROMPT_AGENT_IDS="${DEXJOCO_EVO_PROMPT_AGENT_IDS:-}"
export DEXJOCO_EVO_PROMPT_BACKEND="${DEXJOCO_EVO_PROMPT_BACKEND:-heuristic}"
export DEXJOCO_EVO_PROMPT_ALLOW_HEURISTIC_FALLBACK="${DEXJOCO_EVO_PROMPT_ALLOW_HEURISTIC_FALLBACK:-0}"
export DEXJOCO_EVO_PROMPT_SELECTION_MODE="${DEXJOCO_EVO_PROMPT_SELECTION_MODE:-score}"
export DEXJOCO_EVO_PROMPT_EVAL_EACH_ROUND="${DEXJOCO_EVO_PROMPT_EVAL_EACH_ROUND:-0}"
export DEXJOCO_ACP_SUFFIX="${DEXJOCO_ACP_SUFFIX:- Use the high-advantage successful strategy.}"

export DEXJOCO_RECAP_INCLUDE_FAILURES="${DEXJOCO_RECAP_INCLUDE_FAILURES:-0}"
export DEXJOCO_RECAP_LABEL_WITH_VALUE="${DEXJOCO_RECAP_LABEL_WITH_VALUE:-1}"
export DEXJOCO_RECAP_VALUE_EPOCHS="${DEXJOCO_RECAP_VALUE_EPOCHS:-8}"
export DEXJOCO_RECAP_VALUE_BATCH_SIZE="${DEXJOCO_RECAP_VALUE_BATCH_SIZE:-64}"
export DEXJOCO_RECAP_VALUE_EVAL_BATCH_SIZE="${DEXJOCO_RECAP_VALUE_EVAL_BATCH_SIZE:-128}"
export DEXJOCO_RECAP_VALUE_IMAGE_SIZE="${DEXJOCO_RECAP_VALUE_IMAGE_SIZE:-96}"
export DEXJOCO_RECAP_VALUE_LR="${DEXJOCO_RECAP_VALUE_LR:-0.0003}"
export DEXJOCO_RECAP_VALUE_N_STEP="${DEXJOCO_RECAP_VALUE_N_STEP:-50}"
export DEXJOCO_RECAP_VALUE_POSITIVE_RATIO="${DEXJOCO_RECAP_VALUE_POSITIVE_RATIO:-0.2}"
export DEXJOCO_RECAP_VALUE_C_FAIL_COEF="${DEXJOCO_RECAP_VALUE_C_FAIL_COEF:-1.0}"
export DEXJOCO_RECAP_VALUE_EXACT_TOP_K="${DEXJOCO_RECAP_VALUE_EXACT_TOP_K:-1}"
export DEXJOCO_RECAP_VALUE_POSITIVE_SUCCESS_ONLY="${DEXJOCO_RECAP_VALUE_POSITIVE_SUCCESS_ONLY:-1}"

export OPENPI_RECAP_BASE_REPEAT="${OPENPI_RECAP_BASE_REPEAT:-0}"
export OPENPI_RECAP_POSITIVE_REPEAT="$DEXJOCO_EVO_POSITIVE_REPEAT"
export OPENPI_RECAP_INDICATOR_FIELD="${OPENPI_RECAP_INDICATOR_FIELD:-acp_indicator}"
export DEXJOCO_RECAP_FSDP_DEVICES="${DEXJOCO_RECAP_FSDP_DEVICES:-1}"
export DEXJOCO_RECAP_NUM_WORKERS="${DEXJOCO_RECAP_NUM_WORKERS:-0}"
export DEXJOCO_RECAP_WARMUP_STEPS="${DEXJOCO_RECAP_WARMUP_STEPS:-100}"
export DEXJOCO_RECAP_SAVE_INTERVAL="${DEXJOCO_RECAP_SAVE_INTERVAL:-400}"

RUN_ROOT="${RUN_ROOT:-$(_dexjoco_run_root)}"
DEXJOCO_DIR="${DEXJOCO_DIR:-$RUN_ROOT/dexjoco-src}"

if [[ "$DEXJOCO_EVO_MERGE_POOL" == "1" ]]; then
  mode="pool"
else
  mode="nopool"
fi

OUT_BASE="$OUTPUT_DIR/dexjoco_click_mouse_evorl_multiround_${mode}"
mkdir -p "$OUT_BASE"
MULTI_SUMMARY="$OUT_BASE/multiround_summary.tsv"
PROMPT_HISTORY="$OUT_BASE/prompt_history.jsonl"
echo -e "round\tmode\tcollect_seed\tcollect_prompt\tcollect_episodes\tpool_inputs\tcheckpoint_step\teval_status\tsuccesses\tepisodes" | tee "$MULTI_SUMMARY"

prev_policy_dir="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK"
prev_pretrained_model_path="../checkpoints/pi05_dexjoco_ckpt/$DEXJOCO_TASK/params"
pool_inputs=()

if [[ -n "$DEXJOCO_EVO_INITIAL_POOL_INPUTS" ]]; then
  read -r -a initial_pool_inputs <<< "$DEXJOCO_EVO_INITIAL_POOL_INPUTS"
  pool_inputs+=("${initial_pool_inputs[@]}")
  echo "[evorl] loaded initial data pool inputs: ${initial_pool_inputs[*]}"
fi

if [[ "$DEXJOCO_EVO_INITIAL_COLLECT_EPISODES" -gt 0 ]]; then
  initial_tag="d00"
  initial_exp_name="recap_evorl_${mode}_${initial_tag}"
  initial_data_prefix="evorl_${mode}_${initial_tag}"
  initial_output_name="dexjoco_click_mouse_evorl_multiround_${mode}/initial_${initial_tag}"
  initial_npz="$RUN_ROOT/${initial_data_prefix}_collected_rollouts.npz"
  initial_shard_npzs=()

  echo "[evorl] collecting initial data pool D0 episodes=$DEXJOCO_EVO_INITIAL_COLLECT_EPISODES seed=$DEXJOCO_EVO_INITIAL_COLLECT_SEED shard_episodes=$DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES policy=$prev_policy_dir"
  remaining="$DEXJOCO_EVO_INITIAL_COLLECT_EPISODES"
  shard=0
  while [[ "$remaining" -gt 0 ]]; do
    if [[ "$DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES" -gt 0 && "$DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES" -lt "$remaining" ]]; then
      shard_episodes="$DEXJOCO_EVO_INITIAL_COLLECT_SHARD_EPISODES"
    else
      shard_episodes="$remaining"
    fi
    shard_tag="$(printf '%s_s%02d' "$initial_tag" "$shard")"
    shard_exp_name="recap_evorl_${mode}_${shard_tag}"
    shard_data_prefix="evorl_${mode}_${shard_tag}"
    shard_output_name="dexjoco_click_mouse_evorl_multiround_${mode}/initial_${shard_tag}"
    shard_seed=$((DEXJOCO_EVO_INITIAL_COLLECT_SEED + shard))

    echo "[evorl] collecting D0 shard=$shard episodes=$shard_episodes seed=$shard_seed"
    DEXJOCO_COLLECT_EPISODES="$shard_episodes" \
    DEXJOCO_COLLECT_SEED="$shard_seed" \
    DEXJOCO_RECAP_COLLECT_PROMPT_MODE="base" \
    DEXJOCO_RECAP_OUTPUT_NAME="$shard_output_name" \
    DEXJOCO_RECAP_DATA_PREFIX="$shard_data_prefix" \
    DEXJOCO_RECAP_EXP_NAME="$shard_exp_name" \
    DEXJOCO_RECAP_ROLLOUT_POLICY_DIR="$prev_policy_dir" \
    DEXJOCO_RECAP_PRETRAINED_MODEL_PATH="$prev_pretrained_model_path" \
    DEXJOCO_RECAP_POOL_INPUTS="" \
    DEXJOCO_RECAP_SKIP_EVAL="1" \
    DEXJOCO_RECAP_COLLECT_ONLY="1" \
      bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

    initial_shard_npzs+=("$RUN_ROOT/${shard_data_prefix}_collected_rollouts.npz")
    remaining=$((remaining - shard_episodes))
    shard=$((shard + 1))
  done

  echo "[evorl] merging D0 shards into $initial_npz: ${initial_shard_npzs[*]}"
  python "$EXP_DIR/scripts/dexjoco_merge_rollout_npz.py" \
    --output "$initial_npz" \
    --summary-output "$OUT_BASE/initial_${initial_tag}_pool.summary.json" \
    "${initial_shard_npzs[@]}"
  pool_inputs+=("$initial_npz")
  echo -e "0\t$mode\t$DEXJOCO_EVO_INITIAL_COLLECT_SEED\tbase\t$DEXJOCO_EVO_INITIAL_COLLECT_EPISODES\t${#pool_inputs[@]}\tNA\tinitial_pool\tNA\tNA" | tee -a "$MULTI_SUMMARY"
fi

for round in $(seq 1 "$DEXJOCO_EVO_ROUNDS"); do
  round_tag="$(printf 'r%02d' "$round")"
  exp_name="recap_evorl_${mode}_${round_tag}"
  data_prefix="evorl_${mode}_${round_tag}"
  output_name="dexjoco_click_mouse_evorl_multiround_${mode}/round_${round_tag}"
  collect_seed=$((DEXJOCO_EVAL_SEED + round - 1))

  if [[ "$round" -eq 1 ]]; then
    collect_prompt_mode="base"
  else
    collect_prompt_mode="acp"
  fi

  if [[ "$round" -lt "$DEXJOCO_EVO_ROUNDS" && ! ( "$DEXJOCO_EVO_PROMPT_AGENTS" == "1" && "$DEXJOCO_EVO_PROMPT_EVAL_EACH_ROUND" == "1" ) ]]; then
    skip_eval="1"
  else
    skip_eval="0"
  fi

  if [[ "$DEXJOCO_EVO_MERGE_POOL" == "1" && "${#pool_inputs[@]}" -gt 0 ]]; then
    pool_input_string="${pool_inputs[*]}"
  else
    pool_input_string=""
  fi

  prompt_manifest=""
  prompt_suffix_file=""
  openpi_prompt_manifest="${OPENPI_RECAP_ACP_PROMPTS_FILE:-}"
  if [[ "$DEXJOCO_EVO_PROMPT_AGENTS" == "1" ]]; then
    prompt_manifest="$OUT_BASE/prompt_${round_tag}.json"
    prompt_suffix_file="$OUT_BASE/prompt_${round_tag}.suffix.txt"
    prompt_agent_args=(
      --task "$DEXJOCO_TASK"
      --round "$round"
      --base-config "$DEXJOCO_DIR/configs/rand_obj/${DEXJOCO_TASK}.yaml"
      --history "$PROMPT_HISTORY"
      --output "$prompt_manifest"
      --suffix-output "$prompt_suffix_file"
      --backend "$DEXJOCO_EVO_PROMPT_BACKEND"
      --selection-mode "$DEXJOCO_EVO_PROMPT_SELECTION_MODE"
      --seed "$collect_seed"
    )
    if [[ "$DEXJOCO_EVO_PROMPT_ALLOW_HEURISTIC_FALLBACK" == "1" ]]; then
      prompt_agent_args+=(--allow-heuristic-fallback)
    fi
    if [[ -n "$DEXJOCO_EVO_PROMPT_AGENT_IDS" ]]; then
      read -r -a prompt_agent_ids <<< "$DEXJOCO_EVO_PROMPT_AGENT_IDS"
      prompt_agent_args+=(--agent-ids "${prompt_agent_ids[@]}")
    fi
    python "$EXP_DIR/scripts/dexjoco_generate_multiagent_prompts.py" generate "${prompt_agent_args[@]}"
    DEXJOCO_ACP_SUFFIX="$(< "$prompt_suffix_file")"
    openpi_prompt_manifest="$prompt_manifest"
    echo "[evorl] multi-agent prompt manifest=$prompt_manifest suffix=$DEXJOCO_ACP_SUFFIX"
  fi

  echo "[evorl] round=$round/$DEXJOCO_EVO_ROUNDS mode=$mode collect_prompt=$collect_prompt_mode collect_seed=$collect_seed pool_inputs=${#pool_inputs[@]} rollout_policy=$prev_policy_dir pretrained=$prev_pretrained_model_path prompt_agents=$DEXJOCO_EVO_PROMPT_AGENTS"

  DEXJOCO_COLLECT_EPISODES="$DEXJOCO_EVO_COLLECT_EPISODES" \
  DEXJOCO_COLLECT_SEED="$collect_seed" \
  DEXJOCO_RECAP_COLLECT_PROMPT_MODE="$collect_prompt_mode" \
  DEXJOCO_ACP_SUFFIX="$DEXJOCO_ACP_SUFFIX" \
  OPENPI_RECAP_ACP_PROMPTS_FILE="$openpi_prompt_manifest" \
  DEXJOCO_RECAP_OUTPUT_NAME="$output_name" \
  DEXJOCO_RECAP_DATA_PREFIX="$data_prefix" \
  DEXJOCO_RECAP_EXP_NAME="$exp_name" \
  DEXJOCO_RECAP_TRAIN_STEPS="$DEXJOCO_EVO_TRAIN_STEPS" \
  DEXJOCO_RECAP_WARMUP_STEPS="$DEXJOCO_RECAP_WARMUP_STEPS" \
  DEXJOCO_RECAP_SAVE_INTERVAL="$DEXJOCO_RECAP_SAVE_INTERVAL" \
  DEXJOCO_RECAP_ROLLOUT_POLICY_DIR="$prev_policy_dir" \
  DEXJOCO_RECAP_PRETRAINED_MODEL_PATH="$prev_pretrained_model_path" \
  DEXJOCO_RECAP_POOL_INPUTS="$pool_input_string" \
  DEXJOCO_RECAP_POOL_MAX_FRAMES="$DEXJOCO_RECAP_POOL_MAX_FRAMES" \
  DEXJOCO_RECAP_POOL_MAX_EPISODES="${DEXJOCO_RECAP_POOL_MAX_EPISODES:-0}" \
  DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT="$DEXJOCO_RECAP_POOL_KEEP_LAST_INPUT" \
  DEXJOCO_RECAP_SKIP_EVAL="$skip_eval" \
    bash jobs/25_dexjoco_click_mouse_recap_rollout_finetune.sh

  collected_npz="$RUN_ROOT/${data_prefix}_collected_rollouts.npz"
  pool_inputs+=("$collected_npz")

  ckpt_root="$DEXJOCO_DIR/checkpoints/recap_acp_ckpts/$DEXJOCO_TASK/$exp_name"
  step="$(find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)"
  if [[ -z "$step" ]]; then
    echo "[evorl] no checkpoint found in $ckpt_root" >&2
    exit 1
  fi
  prev_policy_dir="$ckpt_root/$step"
  prev_pretrained_model_path="$prev_policy_dir/params"
  if [[ ! -d "$prev_pretrained_model_path" ]]; then
    echo "[evorl] expected checkpoint params not found: $prev_pretrained_model_path" >&2
    find "$prev_policy_dir" -maxdepth 2 -type f -o -type d >&2 || true
    exit 1
  fi

  round_summary="$OUTPUT_DIR/$output_name/summary.tsv"
  eval_status="train_only"
  successes="NA"
  episodes="NA"
  if [[ -f "$round_summary" ]]; then
    line="$(tail -n 1 "$round_summary")"
    eval_status="$(echo "$line" | cut -f2)"
    successes="$(echo "$line" | cut -f3)"
    episodes="$(echo "$line" | cut -f4)"
  fi
  echo -e "$round\t$mode\t$collect_seed\t$collect_prompt_mode\t$DEXJOCO_EVO_COLLECT_EPISODES\t${#pool_inputs[@]}\t$step\t$eval_status\t$successes\t$episodes" | tee -a "$MULTI_SUMMARY"

  if [[ "$DEXJOCO_EVO_PROMPT_AGENTS" == "1" && -n "$prompt_manifest" ]]; then
    python "$EXP_DIR/scripts/dexjoco_generate_multiagent_prompts.py" record-result \
      --manifest "$prompt_manifest" \
      --history "$PROMPT_HISTORY" \
      --mode "$mode" \
      --collect-seed "$collect_seed" \
      --checkpoint-step "$step" \
      --eval-status "$eval_status" \
      --successes "$successes" \
      --episodes "$episodes"
  fi
done

echo "[evorl] DexJoCo click_mouse multi-round ReCap finished: $MULTI_SUMMARY"
