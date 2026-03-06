@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Run from repo root: c:\dev\Powershell\nodus
set "PYTHON=python"
set "SCRIPT=toys_to_survive_development\wav_config_transformer_pipeline.py"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HOME=%USERPROFILE%\.cache\huggingface"

REM Force classifier init from scratch for this iterative launcher (no Berkeley pretrain checkpoint).

REM Stable output dir for iterative resume
set "OUTPUT_DIR=toys_to_survive_development\wav_pipeline_runs\iterative_rgb256"
set "AUTO_RESUME=1"
set "CKPT_AFTER_SEGMENT=1"
set "CHECKPOINT_EVERY_ROUND=1"
set "BASE_SEED=6666"
set "IMAGE_SIZE=256"
set "MAX_FILES=512"
set "LATENT_COUNT=512"
set "LATENT_SECONDS=13.0"
set "SAVE_PREVIEW=0"
set "TRAINING_PREVIEW_TOPK=8"
set "TRAINING_PREVIEW_GEN_SAMPLES=6"
set "TRAINING_PREVIEW_EVERY=1"
set "STAGE_OPENGL_SCALE=1"
set "STAGE_OPENGL_EVERY=1"
set "STAGE_OPENGL_BOOTSTRAP=1"
set "STAGE_OPENGL_REQUIRED=1"

REM Optional WAV source. Leave empty to use latent fallback + reinjection.
set "WAV_ROOT="

REM Enforced render config (bypass config search/resume config).
set "ENFORCE_RENDER_BITMODE=16"
set "ENFORCE_RENDER_USE_CHANNELS=1"
set "ENFORCE_RENDER_MONO_PICK=Mix"
set "ENFORCE_RENDER_WIDTH=256"
set "ENFORCE_RENDER_DOWNSAMPLE=1"
set "ENFORCE_RENDER_COLOR_MODE=Single source -> RGB via sample stride"
set "ENFORCE_RENDER_EMPTY_FILL=0"
set "ENFORCE_RENDER_BIT_LOW=4"
set "ENFORCE_RENDER_BIT_HIGH=11"
set "ENFORCE_RENDER_MAX_POINTS=262144"

REM Gate settings (downstream will be skipped until these are maintained)
set "GATE_GESTATION_LOSS_TARGET=1.2"
set "GATE_GESTATION_MAINTAIN=2"
set "GATE_GESTATION_BATCH_SIZE=64"
set "GATE_TOTAL_TOKEN_SCHEDULE=1"
set "GATE_TOTAL_TOKEN_SCHEDULE_THRESHOLD=0.55"
set "GATE_BERKELEY_MIN=0.8"
set "GATE_BERKELEY_LOSS_TARGET=1.1"
set "GATE_BERKELEY_MAINTAIN=2"
set "GATE_BERKELEY_BATCH_SIZE=64"
set "GATE_TRANS_AFTER_MIN=0.265"
set "GATE_TRANS_GAIN_MIN=0.0005"
set "GATE_TRANS_MAINTAIN=2"

REM Training schedule defaults
set "ORCH_MODE=staged_cgrw"
set "ORCH_CYCLES=8"
set "ORCH_ROUNDS=2"
set "GLOBAL_GRAD_ACCUM_STEPS=4"
set "CLASSIFIER_BATCH_SIZE=32"
set "GEN_EPOCHS_PER_ROUND=5"
set "GEN_STEPS_PER_ROUND=64"
set "GEN_BATCH_SIZE=12"
set "GEN_GRAD_ACCUM_STEPS=3"
set "GEN_CLASSIFIER_BATCH_CAP=6"
set "GEN_Z_DIM=128"
set "GEN_LR=0.0002"
set "DISC_LR=0.0002"
set "GEN_LOSS_ADV_W=0.50"
set "GEN_LOSS_CLS_W=1.50"
set "GEN_LOSS_WAVE_W=0.80"
set "GEN_LOG_EVERY=8"
set "GEN_WAVE_SCORE_MARGIN=0.02"
set "GEN_WAVE_DENOISE_W=0.25"
set "GEN_WAVE_BLEND=0.35"
set "GEN_WAVE_STRENGTH=0.35"
set "GEN_WAVE_STRIDE_SKEW_MAX=0.18"
set "GEN_WAVE_DEGRADE_MODE=blur_stride"
set "GATE_GEN_MIN_TARGET_PROB=0.55"
set "GATE_GEN_MAINTAIN=2"
set "JOINT_GEN_EPOCHS_PER_ROUND=1"
set "JOINT_GEN_STEPS_PER_ROUND=64"
set "JOINT_TRANS_EPOCHS_PER_ROUND=1"
set "JOINT_TRANS_STEPS_PER_ROUND=64"
set "TRANS_EPOCHS_PER_ROUND=5"
set "TRANS_STEPS_PER_ROUND=256"
set "BERKELEY_REFRESH_ROUND_EVERY=1"
set "BERKELEY_REFRESH_EPOCHS=1"
set "BERKELEY_REFRESH_MAX_STEPS=256"
set "BERKELEY_REFRESH_BATCH_SIZE=64"
set "BERKELEY_REFRESH_LOADER_BATCH_SIZE=256"
set "BERKELEY_REFRESH_CACHE_BATCHES=8"
set "BERKELEY_REFRESH_CACHE_DEVICE=cpu"
set "BERKELEY_REFRESH_WORKERS=2"
set "BERKELEY_PAYLOAD_MAX_SAMPLES=2048"
set "BERKELEY_PAYLOAD_SOURCE_ROOT="
set "BERKELEY_PAYLOAD_CACHE_REBUILD=0"
set "BERKELEY_REFRESH_LOG_EVERY=8"
set "TRANS_BATCH_SIZE=2"
set "TRANS_ROUND_BATCH_SIZE=3"
set "TRANS_JOINT_BATCH_SIZE=1"
set "TRANS_EVAL_BATCH_SIZE=2"
set "TRANS_LOG_EVERY=32"
set "TRANS_LOSS_ENTROPY_W=0.60"
set "TRANS_LOSS_HI_W=0.50"
set "TRANS_LOSS_WAVE_W=1.00"
set "TRANS_LOSS_LO_W=0.08"
set "TRANS_LOSS_SCORE_TARGET_W=2.00"
set "TRANS_SCORE_MARGIN=0.02"
set "LR_SINE_CYCLES=0"
set "LR_SINE_FREQUENCY=0.0"
set "LR_SINE_TAIL_FRACTION=0.0"
set "LR_SINE_MIN_SCALE=1.0"
set "TRANS_DEGRADE_VEC_WINDOW=2"
set "TRANS_DEGRADE_DOMAIN=bitwindow"
set "TRANS_DEGRADE_MODE=blur_stride"
set "TRANS_DEGRADE_STRIDE_SKEW_MAX=0.18"
set "LATENT_IMPRINT_STEPS=12"
set "LATENT_IMPRINT_LR=0.08"
set "LATENT_IMPRINT_L2=0.02"
set "LATENT_IMPRINT_MIN_CLASSES=2"
set "LATENT_IMPRINT_MAX_CLASSES=4"
set "WAVE_CLS_TRAIN_SAMPLES=1024"
set "WAVE_CLS_VAL_SAMPLES=384"
set "WAVE_CLS_EPOCHS_PER_ROUND=1"
set "WAVE_CLS_BATCH_SIZE=64"
set "WAVE_CLS_LOG_EVERY=8"
set "WAVE_CLS_SEMANTIC_MIX_NOISE_PROB=0.10"
set "WAVE_CLS_SEMANTIC_MIX_NOISE_STD=0.03"
set "WAVE_CLS_SEMANTIC_MIX_BLEND_MIN=0.10"
set "WAVE_CLS_SEMANTIC_MIX_BLEND_MAX=0.35"
set "CLASSIFIER_SUBSET_FRESH_ROUND_SAMPLING=1"
set "LIBRARY_MAX_PER_ROUND=96"
set "TRANS_ACCEPTED_PRELOAD_MAX=1024"
set "TRANS_ACCEPTED_APPEND_MAX_PER_ROUND=96"
set "LABEL_EMBED_BACKEND=sentence_transformers"
set "LABEL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2"
set "LABEL_EMBED_DIM=384"
set "LABEL_EMBED_TEMP=10.0"
set "LABEL_QUERY_TEXTS="
set "LABEL_QUERY_TOPK=5"
set "SEMANTIC_VOCAB_EXTRA_TEXTS="
set "SEMANTIC_VOCAB_EXTRA_JSON="
set "SEMANTIC_VOCAB_EXTRA_SLOTS=48"
set "SEMANTIC_VOCAB_CHURN_ENABLED=1"
set "SEMANTIC_VOCAB_CHURN_REPLACE_PER_CYCLE=2"
set "SEMANTIC_VOCAB_REGURGITATED_CHURN_ENABLED=1"
set "SEMANTIC_VOCAB_REGURGITATED_CHURN_PROB=0.35"
set "SEMANTIC_VOCAB_REGURGITATED_CHURN_LIFETIME=3"
set "SEMANTIC_VOCAB_REGURGITATED_CHURN_MAX_PER_CYCLE=4"
set "SEMANTIC_VOCAB_REGURGITATED_CHURN_CAPACITY=512"
set "SEMANTIC_VOCAB_TEXT_CONDITION_TEMPERATURE=8.0"
set "SEMANTIC_VOCAB_TEXT_CONDITION_TOPK=0"
set "SEMANTIC_VOCAB_UNKNOWN_LABEL_ROWS_PER_CYCLE=8"
set "SEMANTIC_VOCAB_BOOTSTRAP_ORIGIN_LABEL=internal bootstrap root vocab"
set "SEMANTIC_VOCAB_AUTO_SYMBOL_POOL=1"
set "SEMANTIC_VOCAB_SYMBOL_POOL_ROOT=toys_to_survive_development\data\semantic_symbol_pool"
set "SEMANTIC_VOCAB_SYMBOL_SAMPLES_PER_TERM=1"
set "SEMANTIC_VOCAB_GESTATION_TRAIN_TARGET_SAMPLES=8000"
set "SEMANTIC_VOCAB_GESTATION_VAL_TARGET_SAMPLES=1600"
set "SEMANTIC_VOCAB_SYMBOL_INCLUDE_PICTOGRAMS=0"
set "SEMANTIC_VOCAB_REFERENCE_FLASHCARDS=1"
set "SEMANTIC_VOCAB_REFERENCE_FLASHCARDS_PER_TERM=2"
set "GD_VOCAB_LIBRARY_ENABLED=1"
set "GD_VOCAB_LIBRARY_AUTOLOAD=1"
set "GD_VOCAB_LIBRARY_DIR="
set "WAVE_ZERO_SHOT_QUERY_TEXTS="
set "WAVE_ZERO_SHOT_TOPK=3"
set "WAVE_ZERO_SHOT_MAX_SAMPLES=128"
set "WAVE_ZERO_SHOT_REPORT_SAMPLES=8"
set "CLS_BASE_CH=96"
set "CLS_MAX_CH=512"
set "CLS_CONTEXT_BLOCKS=12"
set "CLS_CONTEXT_DROPOUT=0.05"
set "TRANS_PATCH_SIZE=32"
set "TRANS_D_MODEL=256"
set "TRANS_NHEAD=8"
set "TRANS_NUM_LAYERS=6"
set "TRANS_FF_MULT=3"
set "TRANS_DROPOUT=0.10"
set "GEN_BASE_CH=160"
set "GEN_DEPTH=7"
set "GEN_MIN_CH=24"
set "DISC_BASE_CH=128"
set "DISC_DEPTH=6"
set "DISC_MAX_CH=768"
set "DISC_STEPS_PER_GEN_STEP=2"
set "FAKE_SENTINEL_LABEL=GAN image"
set "GEN_FAKE_FEEDBACK_EPOCHS=1"
set "GEN_FAKE_FEEDBACK_STEPS_PER_ROUND=48"
set "GEN_FAKE_FEEDBACK_BATCH_SIZE=32"
set "GEN_FAKE_FEEDBACK_LR=0.0006"
set "GEN_FAKE_FEEDBACK_WEIGHT_DECAY=0.0001"
set "GEN_FAKE_FEEDBACK_VECTOR_W=1.00"
set "GEN_FAKE_FEEDBACK_COND_W=0.35"
set "GEN_FAKE_FEEDBACK_DISC_CONF_TEMP=1.00"
set "GEN_FAKE_FEEDBACK_DISC_CONF_FLOOR=0.25"
set "GEN_FAKE_FEEDBACK_LOG_EVERY=0"
set "ENDLESS_MODE=1"
set "ENDLESS_MAX_RUNS=0"
set "ENDLESS_SEED_STRIDE=9973"
set "ENDLESS_SLEEP_SECONDS=2"
set "GUI_STOP_EXIT_CODE=42"
REM Optional runtime overrides applied before each endless run.
REM Example file contents:
REM   set "GATE_BERKELEY_LOSS_TARGET=0.25"
REM   set "GATE_BERKELEY_MIN=0.8"
set "RUNTIME_OVERRIDES_FILE=toys_to_survive_development\run_iterative_wav_pipeline.overrides.bat"

set "WAV_ARG="
if not "%WAV_ROOT%"=="" set "WAV_ARG=--wav-root ""%WAV_ROOT%"""
set "RESUME_ARG="
if "%AUTO_RESUME%"=="1" set "RESUME_ARG=--auto-resume"
set "CKPT_SEGMENT_ARG="
if "%CKPT_AFTER_SEGMENT%"=="1" set "CKPT_SEGMENT_ARG=--checkpoint-after-training-segment"
set "LABEL_QUERY_ARG="
if not "%LABEL_QUERY_TEXTS%"=="" set "LABEL_QUERY_ARG=--label-query-texts ""%LABEL_QUERY_TEXTS%"" --label-query-topk %LABEL_QUERY_TOPK%"
set "SEMANTIC_VOCAB_ARG="
if not "%SEMANTIC_VOCAB_EXTRA_TEXTS%"=="" set "SEMANTIC_VOCAB_ARG=--semantic-vocab-extra-texts ""%SEMANTIC_VOCAB_EXTRA_TEXTS%"""
if not "%SEMANTIC_VOCAB_EXTRA_JSON%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-extra-json ""%SEMANTIC_VOCAB_EXTRA_JSON%"""
if not "%SEMANTIC_VOCAB_EXTRA_SLOTS%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-extra-slots %SEMANTIC_VOCAB_EXTRA_SLOTS%"
if "%SEMANTIC_VOCAB_CHURN_ENABLED%"=="1" (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-churn-enabled"
) else (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --no-semantic-vocab-churn-enabled"
)
if not "%SEMANTIC_VOCAB_CHURN_REPLACE_PER_CYCLE%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-churn-replace-per-cycle %SEMANTIC_VOCAB_CHURN_REPLACE_PER_CYCLE%"
if "%SEMANTIC_VOCAB_REGURGITATED_CHURN_ENABLED%"=="1" (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-regurgitated-churn-enabled"
) else (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --no-semantic-vocab-regurgitated-churn-enabled"
)
if not "%SEMANTIC_VOCAB_REGURGITATED_CHURN_PROB%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-regurgitated-churn-prob %SEMANTIC_VOCAB_REGURGITATED_CHURN_PROB%"
if not "%SEMANTIC_VOCAB_REGURGITATED_CHURN_LIFETIME%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-regurgitated-churn-lifetime %SEMANTIC_VOCAB_REGURGITATED_CHURN_LIFETIME%"
if not "%SEMANTIC_VOCAB_REGURGITATED_CHURN_MAX_PER_CYCLE%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-regurgitated-churn-max-per-cycle %SEMANTIC_VOCAB_REGURGITATED_CHURN_MAX_PER_CYCLE%"
if not "%SEMANTIC_VOCAB_REGURGITATED_CHURN_CAPACITY%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-regurgitated-churn-capacity %SEMANTIC_VOCAB_REGURGITATED_CHURN_CAPACITY%"
if not "%SEMANTIC_VOCAB_TEXT_CONDITION_TEMPERATURE%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-text-condition-temperature %SEMANTIC_VOCAB_TEXT_CONDITION_TEMPERATURE%"
if not "%SEMANTIC_VOCAB_TEXT_CONDITION_TOPK%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-text-condition-topk %SEMANTIC_VOCAB_TEXT_CONDITION_TOPK%"
if not "%SEMANTIC_VOCAB_UNKNOWN_LABEL_ROWS_PER_CYCLE%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-unknown-label-rows-per-cycle %SEMANTIC_VOCAB_UNKNOWN_LABEL_ROWS_PER_CYCLE%"
if "%SEMANTIC_VOCAB_AUTO_SYMBOL_POOL%"=="1" (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-auto-symbol-pool"
) else (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --no-semantic-vocab-auto-symbol-pool"
)
if not "%SEMANTIC_VOCAB_SYMBOL_POOL_ROOT%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-symbol-pool-root ""%SEMANTIC_VOCAB_SYMBOL_POOL_ROOT%"""
if not "%SEMANTIC_VOCAB_SYMBOL_SAMPLES_PER_TERM%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-symbol-samples-per-term %SEMANTIC_VOCAB_SYMBOL_SAMPLES_PER_TERM%"
if not "%SEMANTIC_VOCAB_GESTATION_TRAIN_TARGET_SAMPLES%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-gestation-train-target-samples %SEMANTIC_VOCAB_GESTATION_TRAIN_TARGET_SAMPLES%"
if not "%SEMANTIC_VOCAB_GESTATION_VAL_TARGET_SAMPLES%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-gestation-val-target-samples %SEMANTIC_VOCAB_GESTATION_VAL_TARGET_SAMPLES%"
if "%SEMANTIC_VOCAB_SYMBOL_INCLUDE_PICTOGRAMS%"=="1" (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-symbol-include-pictograms"
) else (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --no-semantic-vocab-symbol-include-pictograms"
)
if "%SEMANTIC_VOCAB_REFERENCE_FLASHCARDS%"=="1" (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-reference-flashcards"
) else (
  set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --no-semantic-vocab-reference-flashcards"
)
if not "%SEMANTIC_VOCAB_REFERENCE_FLASHCARDS_PER_TERM%"=="" set "SEMANTIC_VOCAB_ARG=%SEMANTIC_VOCAB_ARG% --semantic-vocab-reference-flashcards-per-term %SEMANTIC_VOCAB_REFERENCE_FLASHCARDS_PER_TERM%"
set "GD_VOCAB_ARG="
if "%GD_VOCAB_LIBRARY_ENABLED%"=="1" (
  set "GD_VOCAB_ARG=%GD_VOCAB_ARG% --gd-vocab-library-enabled"
) else (
  set "GD_VOCAB_ARG=%GD_VOCAB_ARG% --no-gd-vocab-library-enabled"
)
if "%GD_VOCAB_LIBRARY_AUTOLOAD%"=="1" (
  set "GD_VOCAB_ARG=%GD_VOCAB_ARG% --gd-vocab-library-autoload"
) else (
  set "GD_VOCAB_ARG=%GD_VOCAB_ARG% --no-gd-vocab-library-autoload"
)
if not "%GD_VOCAB_LIBRARY_DIR%"=="" set "GD_VOCAB_ARG=%GD_VOCAB_ARG% --gd-vocab-library-dir ""%GD_VOCAB_LIBRARY_DIR%"""
set "WAVE_ZERO_SHOT_ARG="
if not "%WAVE_ZERO_SHOT_QUERY_TEXTS%"=="" set "WAVE_ZERO_SHOT_ARG=--wave-zero-shot-query-texts ""%WAVE_ZERO_SHOT_QUERY_TEXTS%"""
set "CLASSIFIER_SUBSET_REFRESH_ARG="
if "%CLASSIFIER_SUBSET_FRESH_ROUND_SAMPLING%"=="1" (
  set "CLASSIFIER_SUBSET_REFRESH_ARG=--classifier-subset-fresh-round-sampling"
) else (
  set "CLASSIFIER_SUBSET_REFRESH_ARG=--no-classifier-subset-fresh-round-sampling"
)
set "BERKELEY_PAYLOAD_SOURCE_ARG="
if not "%BERKELEY_PAYLOAD_SOURCE_ROOT%"=="" set "BERKELEY_PAYLOAD_SOURCE_ARG=--berkeley-payload-source-root ""%BERKELEY_PAYLOAD_SOURCE_ROOT%"""
set "BERKELEY_PAYLOAD_CACHE_REBUILD_ARG=--no-berkeley-payload-cache-rebuild"
if "%BERKELEY_PAYLOAD_CACHE_REBUILD%"=="1" set "BERKELEY_PAYLOAD_CACHE_REBUILD_ARG=--berkeley-payload-cache-rebuild"
set "GATE_TOTAL_TOKEN_SCHEDULE_ARG=--gate-total-token-schedule-enabled"
if not "%GATE_TOTAL_TOKEN_SCHEDULE%"=="1" set "GATE_TOTAL_TOKEN_SCHEDULE_ARG=--no-gate-total-token-schedule-enabled"

if not exist "%SCRIPT%" (
  echo Missing script: %SCRIPT%
  exit /b 1
)

set "LOCAL_ST_MODEL_SNAPSHOTS=%HF_HOME%\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots"
set "LOCAL_ST_MODEL_DIR="
for /d %%D in ("%LOCAL_ST_MODEL_SNAPSHOTS%\*") do (
  set "LOCAL_ST_MODEL_DIR=%%~fD"
  goto :st_model_found
)
:st_model_found
if not defined LOCAL_ST_MODEL_DIR (
  echo [launcher] Missing local sentence-transformers cache: %LOCAL_ST_MODEL_SNAPSHOTS%
  echo [launcher] Offline mode is ON. Download all-MiniLM-L6-v2 once, then rerun.
  exit /b 1
)
set "LABEL_EMBED_MODEL=!LOCAL_ST_MODEL_DIR!"
echo [launcher] Local embedding model: !LABEL_EMBED_MODEL!

echo [launcher] Berkeley pretrain disabled; classifier init starts from scratch.

if /I "%ORCH_MODE%"=="staged_cgrw" (
  echo [launcher] ORCH_MODE=%ORCH_MODE%: runs C/G/R/W with wave feedback into upstream gates/loss.
  echo [launcher] Note: this is the integrated closed-loop mode for C/G/D/R/W co-training.
) else (
  if /I "%ORCH_MODE%"=="staged_cgr" (
    echo [launcher] ORCH_MODE=%ORCH_MODE%: runs C/G/R and optional J stages.
    echo [launcher] Note: wave-classifier training and wave zero-shot reporting run only in non-staged modes.
  ) else (
    echo [launcher] ORCH_MODE=%ORCH_MODE%: runs C/R/W stages.
    echo [launcher] Note: generator/discriminator and fake-feedback updates run only in staged_cgr or staged_cgrw.
  )
)

set /a RUN_INDEX=0

:run_loop
set /a RUN_INDEX+=1
set /a RUN_SEED=%BASE_SEED% + ((RUN_INDEX - 1) * %ENDLESS_SEED_STRIDE%)
call :apply_runtime_overrides
echo.
echo [launcher] run !RUN_INDEX! seed=!RUN_SEED! endless=%ENDLESS_MODE%
echo [launcher] gate_gestation_loss_target=!GATE_GESTATION_LOSS_TARGET!
echo [launcher] gate_berkeley_loss_target=!GATE_BERKELEY_LOSS_TARGET!
call :run_pipeline !RUN_SEED!
set "RUN_RC=!ERRORLEVEL!"
if "!RUN_RC!"=="%GUI_STOP_EXIT_CODE%" (
  echo.
  echo [launcher] GUI close requested by pipeline, exit !RUN_RC!; stopping run loop without restart.
  goto all_done
)
if not "!RUN_RC!"=="0" (
  echo.
  echo Pipeline run !RUN_INDEX! failed with errorlevel !RUN_RC!.
  exit /b !RUN_RC!
)

if "%ENDLESS_MODE%"=="1" (
  if not "%ENDLESS_MAX_RUNS%"=="0" if !RUN_INDEX! GEQ %ENDLESS_MAX_RUNS% goto all_done
  if %ENDLESS_SLEEP_SECONDS% GTR 0 timeout /t %ENDLESS_SLEEP_SECONDS% /nobreak >nul
  goto run_loop
)

goto all_done

:apply_runtime_overrides
if exist "%RUNTIME_OVERRIDES_FILE%" (
  call "%RUNTIME_OVERRIDES_FILE%"
  set "OVR_RC=!ERRORLEVEL!"
  if not "!OVR_RC!"=="0" (
    echo [launcher] runtime override load failed with errorlevel !OVR_RC!: %RUNTIME_OVERRIDES_FILE%
  ) else (
    echo [launcher] runtime overrides loaded: %RUNTIME_OVERRIDES_FILE%
  )
)
exit /b 0

:run_pipeline
set "RUN_SEED=%~1"
%PYTHON% %SCRIPT% ^
  --objective-mode berkeley_multilabel ^
  --output-dir "%OUTPUT_DIR%" %RESUME_ARG% ^
  --checkpoint-every-round %CHECKPOINT_EVERY_ROUND% %CKPT_SEGMENT_ARG% ^
  --seed %RUN_SEED% ^
  --device cuda:0 ^
  --amp ^
  --amp-dtype bfloat16 ^
  --channels-last ^
  --grad-accum-steps %GLOBAL_GRAD_ACCUM_STEPS% ^
  --no-loader-persistent-workers ^
  --loader-prefetch-factor 2 ^
  --max-files %MAX_FILES% ^
  --image-size %IMAGE_SIZE% ^
  --berkeley-image-size %IMAGE_SIZE% ^
  --save-preview %SAVE_PREVIEW% ^
  --no-training-preview-enabled ^
  --training-preview-topk %TRAINING_PREVIEW_TOPK% ^
  --training-preview-generator-samples %TRAINING_PREVIEW_GEN_SAMPLES% ^
  --training-preview-every-round %TRAINING_PREVIEW_EVERY% ^
  --stage-opengl-preview-enabled ^
  --stage-opengl-preview-bootstrap ^
  --stage-opengl-preview-required ^
  --stage-opengl-preview-scale %STAGE_OPENGL_SCALE% ^
  --stage-opengl-preview-every-round %STAGE_OPENGL_EVERY% ^
  --enforce-render-config ^
  --enforce-render-bitmode %ENFORCE_RENDER_BITMODE% ^
  --enforce-render-use-channels %ENFORCE_RENDER_USE_CHANNELS% ^
  --enforce-render-mono-pick "%ENFORCE_RENDER_MONO_PICK%" ^
  --enforce-render-width %ENFORCE_RENDER_WIDTH% ^
  --enforce-render-downsample %ENFORCE_RENDER_DOWNSAMPLE% ^
  --enforce-render-color-mode "%ENFORCE_RENDER_COLOR_MODE%" ^
  --enforce-render-empty-fill %ENFORCE_RENDER_EMPTY_FILL% ^
  --enforce-render-bitmask-enable ^
  --enforce-render-bitmask-low %ENFORCE_RENDER_BIT_LOW% ^
  --enforce-render-bitmask-high %ENFORCE_RENDER_BIT_HIGH% ^
  --enforce-render-max-points %ENFORCE_RENDER_MAX_POINTS% ^
  --config-trials 8 ^
  --search-train-limit 80 ^
  --search-val-limit 40 ^
  --score-threshold 0.20 ^
  --score-w-topk 0.60 ^
  --score-w-cov 0.30 ^
  --score-w-mean 0.10 ^
  --classifier-batch-size %CLASSIFIER_BATCH_SIZE% ^
  --classifier-base-ch %CLS_BASE_CH% ^
  --classifier-max-ch %CLS_MAX_CH% ^
  --classifier-context-blocks %CLS_CONTEXT_BLOCKS% ^
  --classifier-context-dropout %CLS_CONTEXT_DROPOUT% ^
  --label-embeddings ^
  --label-embedding-backend %LABEL_EMBED_BACKEND% ^
  --label-embedding-model "%LABEL_EMBED_MODEL%" ^
  --label-embedding-dim %LABEL_EMBED_DIM% ^
  --label-embedding-temperature %LABEL_EMBED_TEMP% ^
  --patch-size %TRANS_PATCH_SIZE% ^
  --transformer-batch-size %TRANS_BATCH_SIZE% ^
  --transformer-round-batch-size %TRANS_ROUND_BATCH_SIZE% ^
  --joint-transformer-batch-size %TRANS_JOINT_BATCH_SIZE% ^
  --transformer-eval-batch-size %TRANS_EVAL_BATCH_SIZE% ^
  --transformer-d-model %TRANS_D_MODEL% ^
  --transformer-nhead %TRANS_NHEAD% ^
  --transformer-num-layers %TRANS_NUM_LAYERS% ^
  --transformer-ff-mult %TRANS_FF_MULT% ^
  --transformer-dropout %TRANS_DROPOUT% ^
  --orchestration-mode %ORCH_MODE% ^
  --orchestration-cycles %ORCH_CYCLES% ^
  --orchestration-rounds %ORCH_ROUNDS% ^
  --generator-epochs-per-round %GEN_EPOCHS_PER_ROUND% ^
  --generator-steps-per-round %GEN_STEPS_PER_ROUND% ^
  --generator-batch-size %GEN_BATCH_SIZE% ^
  --generator-grad-accum-steps %GEN_GRAD_ACCUM_STEPS% ^
  --generator-classifier-batch-cap %GEN_CLASSIFIER_BATCH_CAP% ^
  --generator-z-dim %GEN_Z_DIM% ^
  --generator-base-ch %GEN_BASE_CH% ^
  --generator-depth %GEN_DEPTH% ^
  --generator-min-ch %GEN_MIN_CH% ^
  --generator-lr %GEN_LR% ^
  --generator-log-every %GEN_LOG_EVERY% ^
  --discriminator-lr %DISC_LR% ^
  --discriminator-base-ch %DISC_BASE_CH% ^
  --discriminator-depth %DISC_DEPTH% ^
  --discriminator-max-ch %DISC_MAX_CH% ^
  --discriminator-steps-per-generator-step %DISC_STEPS_PER_GEN_STEP% ^
  --generator-loss-adv-weight %GEN_LOSS_ADV_W% ^
  --generator-loss-cls-weight %GEN_LOSS_CLS_W% ^
  --generator-loss-wave-weight %GEN_LOSS_WAVE_W% ^
  --generator-wave-score-margin %GEN_WAVE_SCORE_MARGIN% ^
  --generator-wave-denoise-weight %GEN_WAVE_DENOISE_W% ^
  --generator-wave-carrier-blend %GEN_WAVE_BLEND% ^
  --generator-wave-strength %GEN_WAVE_STRENGTH% ^
  --generator-wave-stride-skew-max %GEN_WAVE_STRIDE_SKEW_MAX% ^
  --generator-wave-degrade-mode %GEN_WAVE_DEGRADE_MODE% ^
  --generator-gate-min-target-prob %GATE_GEN_MIN_TARGET_PROB% ^
  --generator-gate-maintain-rounds %GATE_GEN_MAINTAIN% ^
  --generator-fake-feedback-enabled ^
  --fake-image-sentinel-label "%FAKE_SENTINEL_LABEL%" ^
  --generator-fake-feedback-epochs %GEN_FAKE_FEEDBACK_EPOCHS% ^
  --generator-fake-feedback-steps-per-round %GEN_FAKE_FEEDBACK_STEPS_PER_ROUND% ^
  --generator-fake-feedback-batch-size %GEN_FAKE_FEEDBACK_BATCH_SIZE% ^
  --generator-fake-feedback-lr %GEN_FAKE_FEEDBACK_LR% ^
  --generator-fake-feedback-weight-decay %GEN_FAKE_FEEDBACK_WEIGHT_DECAY% ^
  --generator-fake-feedback-vector-weight %GEN_FAKE_FEEDBACK_VECTOR_W% ^
  --generator-fake-feedback-condition-weight %GEN_FAKE_FEEDBACK_COND_W% ^
  --generator-fake-feedback-disc-conf-temperature %GEN_FAKE_FEEDBACK_DISC_CONF_TEMP% ^
  --generator-fake-feedback-disc-conf-floor %GEN_FAKE_FEEDBACK_DISC_CONF_FLOOR% ^
  --generator-fake-feedback-disc-balance-groups ^
  --generator-fake-feedback-include-condition-targets ^
  --generator-fake-feedback-log-every %GEN_FAKE_FEEDBACK_LOG_EVERY% ^
  --joint-enabled ^
  --joint-generator-epochs-per-round %JOINT_GEN_EPOCHS_PER_ROUND% ^
  --joint-generator-steps-per-round %JOINT_GEN_STEPS_PER_ROUND% ^
  --joint-transformer-epochs-per-round %JOINT_TRANS_EPOCHS_PER_ROUND% ^
  --joint-transformer-steps-per-round %JOINT_TRANS_STEPS_PER_ROUND% ^
  --transformer-epochs-per-round %TRANS_EPOCHS_PER_ROUND% ^
  --transformer-steps-per-round %TRANS_STEPS_PER_ROUND% ^
  --transformer-log-every %TRANS_LOG_EVERY% ^
  --no-transformer-visualize-status ^
  --transformer-viz-scale 3 ^
  --transformer-accepted-preload-max %TRANS_ACCEPTED_PRELOAD_MAX% ^
  --transformer-accepted-append-max-per-round %TRANS_ACCEPTED_APPEND_MAX_PER_ROUND% ^
  --wave-cls-epochs-per-round %WAVE_CLS_EPOCHS_PER_ROUND% ^
  --wave-cls-train-samples %WAVE_CLS_TRAIN_SAMPLES% ^
  --wave-cls-val-samples %WAVE_CLS_VAL_SAMPLES% ^
  --wave-cls-batch-size %WAVE_CLS_BATCH_SIZE% ^
  --wave-cls-log-every %WAVE_CLS_LOG_EVERY% ^
  --wave-cls-semantic-mix-noise-prob %WAVE_CLS_SEMANTIC_MIX_NOISE_PROB% ^
  --wave-cls-semantic-mix-noise-std %WAVE_CLS_SEMANTIC_MIX_NOISE_STD% ^
  --wave-cls-semantic-mix-blend-min %WAVE_CLS_SEMANTIC_MIX_BLEND_MIN% ^
  --wave-cls-semantic-mix-blend-max %WAVE_CLS_SEMANTIC_MIX_BLEND_MAX% ^
  --wave-zero-shot-topk %WAVE_ZERO_SHOT_TOPK% ^
  --wave-zero-shot-max-samples %WAVE_ZERO_SHOT_MAX_SAMPLES% ^
  --wave-zero-shot-report-samples %WAVE_ZERO_SHOT_REPORT_SAMPLES% ^
  --wave-cls-lr 0.001 ^
  --wave-cls-accepted-only ^
  --accept-score-threshold 0.40 ^
  --accept-l1-threshold 0.20 ^
  --library-max-per-round %LIBRARY_MAX_PER_ROUND% ^
  --berkeley-refresh-activation-mult 10.0 ^
  --berkeley-refresh-round-every %BERKELEY_REFRESH_ROUND_EVERY% ^
  --berkeley-refresh-batch-size %BERKELEY_REFRESH_BATCH_SIZE% ^
  --berkeley-refresh-loader-batch-size %BERKELEY_REFRESH_LOADER_BATCH_SIZE% ^
  --berkeley-payload-max-samples %BERKELEY_PAYLOAD_MAX_SAMPLES% ^
  %BERKELEY_PAYLOAD_SOURCE_ARG% ^
  %BERKELEY_PAYLOAD_CACHE_REBUILD_ARG% ^
  --berkeley-refresh-cache-batches %BERKELEY_REFRESH_CACHE_BATCHES% ^
  --berkeley-refresh-cache-device %BERKELEY_REFRESH_CACHE_DEVICE% ^
  --berkeley-refresh-workers %BERKELEY_REFRESH_WORKERS% ^
  --berkeley-refresh-epochs %BERKELEY_REFRESH_EPOCHS% ^
  --berkeley-refresh-max-steps %BERKELEY_REFRESH_MAX_STEPS% ^
  --berkeley-refresh-log-every %BERKELEY_REFRESH_LOG_EVERY% ^
  --berkeley-refresh-max-seconds 600 ^
  --lr-sine-cycles %LR_SINE_CYCLES% ^
  --lr-sine-frequency %LR_SINE_FREQUENCY% ^
  --lr-sine-tail-fraction %LR_SINE_TAIL_FRACTION% ^
  --lr-sine-min-scale %LR_SINE_MIN_SCALE% ^
  --gate-gestation-loss-target %GATE_GESTATION_LOSS_TARGET% ^
  --gate-gestation-batch-size %GATE_GESTATION_BATCH_SIZE% ^
  --gate-gestation-maintain-rounds %GATE_GESTATION_MAINTAIN% ^
  --gate-berkeley-min-score %GATE_BERKELEY_MIN% ^
  --gate-berkeley-loss-target %GATE_BERKELEY_LOSS_TARGET% ^
  --gate-berkeley-batch-size %GATE_BERKELEY_BATCH_SIZE% ^
  --gate-berkeley-maintain-rounds %GATE_BERKELEY_MAINTAIN% ^
  %GATE_TOTAL_TOKEN_SCHEDULE_ARG% ^
  --gate-total-token-schedule-threshold %GATE_TOTAL_TOKEN_SCHEDULE_THRESHOLD% ^
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
  --transformer-degrade-domain %TRANS_DEGRADE_DOMAIN% ^
  --transformer-degrade-mode %TRANS_DEGRADE_MODE% ^
  --transformer-degrade-stride-skew-max %TRANS_DEGRADE_STRIDE_SKEW_MAX% ^
  --transformer-degrade-vectorized-window %TRANS_DEGRADE_VEC_WINDOW% ^
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
  --latent-fallback-count %LATENT_COUNT% ^
  --latent-fallback-seconds %LATENT_SECONDS% ^
  --latent-structured-ratio 0.60 ^
  --latent-structured-gain 0.85 ^
  --latent-structured-noise-gain 0.20 ^
  --latent-reinject-ratio 0.85 ^
  --latent-reinject-copy-gain 0.90 ^
  --latent-reinject-noise-gain 0.15 ^
  %CLASSIFIER_SUBSET_REFRESH_ARG% ^
  --semantic-vocab-bootstrap-origin-label "%SEMANTIC_VOCAB_BOOTSTRAP_ORIGIN_LABEL%" ^
  %SEMANTIC_VOCAB_ARG% ^
  %GD_VOCAB_ARG% ^
  %WAVE_ZERO_SHOT_ARG% ^
  %LABEL_QUERY_ARG% ^
  %WAV_ARG%
exit /b %ERRORLEVEL%

:all_done
echo.
echo Pipeline run loop completed.
endlocal
