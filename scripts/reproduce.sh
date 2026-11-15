#!/usr/bin/env bash
# Every forecast, conformal file and result of the paper, from the three built input files in out_planA_native/
# (README steps 1-2 first). Run from the repository root:
#   NJOBS=8 bash scripts/reproduce.sh                 # full run, about 12 h on an 8-core Apple Silicon Mac
#   bash scripts/reproduce.sh --from conformal        # resume at a stage
#   DRY=1 bash scripts/reproduce.sh                   # print the commands only
# Stages, in order: naive lear xgb conformal retune ablation evaluate.
# Environment: PY (python interpreter, default python), NJOBS (workers, default 8; outputs do not depend on it),
# DRY (1 = print only). Each stage logs to logs/<stage>.log: start time, command, output, exit code and
# wall time of every command. The script stops at the first failing command.
# Compatible with the bash 3.2 of macOS.
set -euo pipefail
export LC_NUMERIC=C          # decimal point in the wall times of `time` (forecasts_retune/conformal_wall.csv)

PY=${PY:-python}
NJOBS=${NJOBS:-8}
DRY=${DRY:-0}
LOGS=logs
STAGES="naive lear xgb conformal retune ablation evaluate"
SETTINGS="hourly native interp"
INPUTS="thesis_features_1hour_full.csv thesis_features_interpolated_mp_full.csv thesis_features_15min_mp_full.csv"

die() { echo "reproduce.sh: $*" >&2; exit 1; }

FROM=naive
while [ $# -gt 0 ]; do
  case $1 in
    --from) [ $# -ge 2 ] || die "--from needs a stage: $STAGES"; FROM=$2; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) die "unknown argument $1 (use --from STAGE)" ;;
  esac
done
case " $STAGES " in *" $FROM "*) ;; *) die "unknown stage $FROM (stages: $STAGES)" ;; esac

[ -d esann27 ] && [ -d scripts ] || die "run from the repository root (esann27/ and scripts/ not found)"
for f in $INPUTS; do
  if [ ! -f "out_planA_native/$f" ]; then
    if [ "$DRY" = 1 ]; then echo "# missing: out_planA_native/$f (README step 2)"; else
      die "out_planA_native/$f missing: build the data first (README step 2)"; fi
  fi
done

# run STAGE CMD...: append start time, command, output, exit code and wall time to logs/STAGE.log;
# the wall time in seconds (3 decimals, bash `time`) is left in LAST_WALL
LAST_WALL=0
run() {
  local stage=$1; shift
  if [ "$DRY" = 1 ]; then echo "[$stage] $*"; return 0; fi
  local log="$LOGS/$stage.log" rc=0
  echo "+ [$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$log"
  LAST_WALL=$( { TIMEFORMAT=%3R; time "$@" >> "$log" 2>&1; } 2>&1 ) || rc=$?
  echo "= exit $rc after $LAST_WALL s" >> "$log"
  if [ $rc -ne 0 ]; then die "stage $stage failed (exit $rc): $* (see $log)"; fi
}

started=0
want() {    # want STAGE: true from the --from stage on; opens the stage log
  [ "$1" = "$FROM" ] && started=1
  [ $started = 1 ] || return 1
  if [ "$DRY" != 1 ]; then
    mkdir -p "$LOGS" 2>/dev/null || die "cannot create the log folder $LOGS/"
    [ -w "$LOGS" ] || die "the log folder $LOGS/ is not writable"
    [ ! -e "$LOGS/$1.log" ] || [ -w "$LOGS/$1.log" ] || die "the log file $LOGS/$1.log is not writable"
    echo "== stage $1 start $(date '+%Y-%m-%d %H:%M:%S'), PY=$PY NJOBS=$NJOBS" >> "$LOGS/$1.log"
  fi
  return 0
}

T0=$SECONDS

# 1. naive forecasts
if want naive; then
  for s in $SETTINGS; do
    run naive "$PY" -u -m esann27.run --setting "$s" --model naive
  done
fi

# 2. LEAR with the lambda path, penalty selection (rules stan, bic, pbic), <setting>_lear.parquet = the pbic file
if want lear; then
  for s in $SETTINGS; do
    run lear "$PY" -u -m esann27.run --setting "$s" --model lear --n-jobs "$NJOBS"
  done
  for s in $SETTINGS; do
    for r in stan bic pbic; do
      run lear "$PY" -u -m esann27.select --setting "$s" --rule "$r"
    done
  done
  for s in $SETTINGS; do
    run lear cp "forecasts/${s}_lear_pbic.parquet" "forecasts/${s}_lear.parquet"
  done
fi

# 3. XGBoost with the frozen hyperparameters (esann27/models/hyperparameters_*.json, never re-tuned here)
if want xgb; then
  for s in $SETTINGS; do
    run xgb "$PY" -u -m esann27.run --setting "$s" --model xgb --start 2025-10-01 --end 2026-10-01 --n-jobs "$NJOBS"
  done
fi

# 4. conformal layer: the five regimes (--regime all), ACI-block and the three diagnostic layers
if want conformal; then
  for m in lear xgb; do
    for s in hourly interp native; do
      for r in all aciblock symblock locblock locslot; do
        run conformal "$PY" -u -m esann27.conformal --setting "$s" --model "$m" --regime "$r" \
          --src forecasts --out conformal
      done
    done
  done
fi

# 5. monthly XGBoost re-tuning (robustness check, model xgbm), then ACI-CQR and ACI-block on it, each conformal
#    call timed into forecasts_retune/conformal_wall.csv (read by evaluate for the wall-time table, as in step 5d)
if want retune; then
  for s in interp native; do
    run retune "$PY" -u -m esann27.retune --setting "$s" --n-jobs "$NJOBS" --out forecasts_retune
  done
  CW=forecasts_retune/conformal_wall.csv
  if [ "$DRY" = 1 ]; then echo "[retune] write $CW (setting,regime,wall_s)"; else echo "setting,regime,wall_s" > "$CW"; fi
  for s in interp native; do
    for r in aci aciblock; do
      run retune "$PY" -u -m esann27.conformal --setting "$s" --model xgbm --regime "$r" \
        --src forecasts_retune --out conformal
      [ "$DRY" = 1 ] || echo "$s,$r,$LAST_WALL" >> "$CW"
    done
  done
fi

# 6. native-share ablation (LEAR, 3-30 Sep 2026)
if want ablation; then
  run ablation "$PY" -u -m esann27.ablation --n-jobs "$NJOBS" --fdir forecasts --out results
fi

# 7. evaluation: xgb, lear, xgb (the order of steps 5d and 5e: the final files are in the row order of the paper)
if want evaluate; then
  for m in xgb lear xgb; do
    run evaluate "$PY" -u -m esann27.evaluate --model "$m" --n-jobs "$NJOBS" --fdir forecasts --cdir conformal \
      --rdir forecasts_retune --out results
  done
fi

[ "$DRY" = 1 ] || echo "reproduce.sh done after $(( SECONDS - T0 )) s"
