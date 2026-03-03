@echo off
setlocal EnableExtensions

REM Run from repo root: c:\dev\Powershell\nodus
set "PYTHON=python"
set "SCRIPT=toys_to_survive_development\wav_config_transformer_pipeline.py"

REM Required pretrained Berkeley classifier checkpoint
set "BERKELEY_CKPT=toys_to_survive_development\wav_pipeline_runs\berkeley_pretrain\berkeley_sbd_multilabel_classifier.pt"

REM Stable output dir for iterative resume
set "OUTPUT_DIR=toys_to_survive_development\wav_pipeline_runs\iterative_main"
set "AUTO_RESUME=1"
set "CKPT_AFTER_SEGMENT=1"

REM Optional WAV source. Leave empty to use latent fallback + reinjection.
set "WAV_ROOT="

REM Gate settings (downstream will be skipped until these are maintained)
set "GATE_BERKELEY_MIN=0.285"
set "GATE_BERKELEY_MAINTAIN=2"
set "GATE_TRANS_AFTER_MIN=0.265"
set "GATE_TRANS_GAIN_MIN=0.010"
set "GATE_TRANS_MAINTAIN=2"

REM Training schedule defaults
set "ORCH_CYCLES=8"
set "ORCH_ROUNDS=2"
set "TRANS_EPOCHS_PER_ROUND=50"
set "TRANS_STEPS_PER_ROUND=80"
set "BERKELEY_REFRESH_EPOCHS=2"
set "BERKELEY_REFRESH_MAX_STEPS=128"
set "TRANS_BATCH_SIZE=192"
set "TRANS_LOSS_ENTROPY_W=0.60"
set "TRANS_LOSS_HI_W=0.50"
set "TRANS_LOSS_WAVE_W=1.00"
set "TRANS_LOSS_LO_W=0.08"
set "TRANS_LOSS_SCORE_TARGET_W=2.00"
set "TRANS_SCORE_MARGIN=0.02"
set "LATENT_IMPRINT_STEPS=12"
set "LATENT_IMPRINT_LR=0.08"
set "LATENT_IMPRINT_L2=0.02"
set "LATENT_IMPRINT_MIN_CLASSES=2"
set "LATENT_IMPRINT_MAX_CLASSES=4"
set "WAVE_CLS_TRAIN_SAMPLES=1024"
set "WAVE_CLS_VAL_SAMPLES=384"
set "WAVE_CLS_BATCH_SIZE=96"
set "LIBRARY_MAX_PER_ROUND=96"
set "TRANS_ACCEPTED_PRELOAD_MAX=1024"
set "TRANS_ACCEPTED_APPEND_MAX_PER_ROUND=96"

set "WAV_ARG="
if not "%WAV_ROOT%"=="" set "WAV_ARG=--wav-root ""%WAV_ROOT%"""
set "RESUME_ARG="
if "%AUTO_RESUME%"=="1" set "RESUME_ARG=--auto-resume"
set "CKPT_SEGMENT_ARG="
if "%CKPT_AFTER_SEGMENT%"=="1" set "CKPT_SEGMENT_ARG=--checkpoint-after-training-segment"

if not exist "%SCRIPT%" (
  echo Missing script: %SCRIPT%
  exit /b 1
)

if not exist "%BERKELEY_CKPT%" (
  echo Missing Berkeley checkpoint: %BERKELEY_CKPT%
  echo Train it first with berkeley_sbd_pretrain.py or update BERKELEY_CKPT path.
  exit /b 1
)

%PYTHON% %SCRIPT% ^
  --objective-mode berkeley_multilabel ^
  --classifier-init-ckpt "%BERKELEY_CKPT%" ^
  --output-dir "%OUTPUT_DIR%" %RESUME_ARG% ^
  --checkpoint-every-round 5 %CKPT_SEGMENT_ARG% ^
  --device cuda:0 ^
  --amp ^
  --amp-dtype bfloat16 ^
  --channels-last ^
  --grad-accum-steps 2 ^
  --pin-memory-wave-batches ^
  --cache-wave-cls-dataset-on-device ^
  --cache-wave-streams-on-device ^
  --loader-persistent-workers ^
  --loader-prefetch-factor 4 ^
  --max-files 256 ^
  --config-trials 8 ^
  --search-train-limit 80 ^
  --search-val-limit 40 ^
  --score-threshold 0.20 ^
  --score-w-topk 0.60 ^
  --score-w-cov 0.30 ^
  --score-w-mean 0.10 ^
  --transformer-batch-size %TRANS_BATCH_SIZE% ^
  --orchestration-mode alternating ^
  --orchestration-cycles %ORCH_CYCLES% ^
  --orchestration-rounds %ORCH_ROUNDS% ^
  --transformer-epochs-per-round %TRANS_EPOCHS_PER_ROUND% ^
  --transformer-steps-per-round %TRANS_STEPS_PER_ROUND% ^
  --transformer-log-every 20 ^
  --transformer-visualize-status ^
  --transformer-viz-scale 3 ^
  --transformer-accepted-preload-max %TRANS_ACCEPTED_PRELOAD_MAX% ^
  --transformer-accepted-append-max-per-round %TRANS_ACCEPTED_APPEND_MAX_PER_ROUND% ^
  --wave-cls-epochs-per-round 1 ^
  --wave-cls-train-samples %WAVE_CLS_TRAIN_SAMPLES% ^
  --wave-cls-val-samples %WAVE_CLS_VAL_SAMPLES% ^
  --wave-cls-batch-size %WAVE_CLS_BATCH_SIZE% ^
  --wave-cls-lr 0.001 ^
  --wave-cls-accepted-only ^
  --accept-score-threshold 0.40 ^
  --accept-l1-threshold 0.20 ^
  --library-max-per-round %LIBRARY_MAX_PER_ROUND% ^
  --berkeley-refresh-activation-mult 10.0 ^
  --berkeley-refresh-round-every 1 ^
  --berkeley-refresh-batch-size 512 ^
  --berkeley-refresh-loader-batch-size 256 ^
  --berkeley-refresh-cache-batches 32 ^
  --berkeley-refresh-cache-device cuda ^
  --berkeley-refresh-workers 4 ^
  --berkeley-refresh-epochs %BERKELEY_REFRESH_EPOCHS% ^
  --berkeley-refresh-max-steps %BERKELEY_REFRESH_MAX_STEPS% ^
  --berkeley-refresh-log-every 16 ^
  --berkeley-refresh-max-seconds 600 ^
  --lr-sine-cycles 1.5 ^
  --lr-sine-tail-fraction 0.20 ^
  --gate-berkeley-min-score %GATE_BERKELEY_MIN% ^
  --gate-berkeley-maintain-rounds %GATE_BERKELEY_MAINTAIN% ^
  --gate-transformer-min-score-after %GATE_TRANS_AFTER_MIN% ^
  --gate-transformer-min-gain %GATE_TRANS_GAIN_MIN% ^
  --gate-transformer-maintain-rounds %GATE_TRANS_MAINTAIN% ^
  --max-delta 0.80 ^
  --transformer-degrade-inputs ^
  --transformer-degrade-min-strength 0.35 ^
  --transformer-degrade-max-strength 0.85 ^
  --transformer-degrade-noise-std-min 0.01 ^
  --transformer-degrade-noise-std-max 0.14 ^
  --transformer-degrade-dropout-max 0.20 ^
  --transformer-degrade-quant-bits-min 4 ^
  --transformer-degrade-quant-bits-max 10 ^
  --transformer-loss-entropy-weight %TRANS_LOSS_ENTROPY_W% ^
  --transformer-loss-high-bit-weight %TRANS_LOSS_HI_W% ^
  --transformer-loss-wave-l1-weight %TRANS_LOSS_WAVE_W% ^
  --transformer-loss-low-bit-weight %TRANS_LOSS_LO_W% ^
  --transformer-loss-score-target-weight %TRANS_LOSS_SCORE_TARGET_W% ^
  --transformer-loss-score-target-margin %TRANS_SCORE_MARGIN% ^
  --latent-berkeley-imprint-mix-targets ^
  --latent-berkeley-imprint-steps %LATENT_IMPRINT_STEPS% ^
  --latent-berkeley-imprint-lr %LATENT_IMPRINT_LR% ^
  --latent-berkeley-imprint-l2 %LATENT_IMPRINT_L2% ^
  --latent-berkeley-imprint-min-target-classes %LATENT_IMPRINT_MIN_CLASSES% ^
  --latent-berkeley-imprint-max-target-classes %LATENT_IMPRINT_MAX_CLASSES% ^
  --latent-fallback-count 256 ^
  --latent-structured-ratio 0.60 ^
  --latent-structured-gain 0.85 ^
  --latent-structured-noise-gain 0.20 ^
  --latent-reinject-ratio 0.85 ^
  --latent-reinject-copy-gain 0.90 ^
  --latent-reinject-noise-gain 0.15 ^
  %WAV_ARG%

if errorlevel 1 (
  echo.
  echo Pipeline run failed with errorlevel %errorlevel%.
  exit /b %errorlevel%
)

echo.
echo Pipeline run completed. Re-run this same script to continue from checkpoint.
endlocal
