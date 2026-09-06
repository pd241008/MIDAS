#!/bin/bash
# On-Pi real-TFLite phase-split latency measurement.
# Runs phase_latency against the real production TFLite classifier and
# writes the parsed per-phase summary to results/pi/phase_split_latency_real_ffi_pi.json
cd ~/midas
export EDGE_CONFIG=configs/edge_config.json
export EDGE_MODEL_PATH=models/classifier_float32.tflite
export LD_LIBRARY_PATH=$HOME/midas/vendor/tflite/aarch64
export PHASE_BENIGN=800
export PHASE_ADV=800
export PHASE_SEGMENTS=16
./target/release/phase_latency > /tmp/pl_pi_full.json 2>/dev/null
tail -n +2 /tmp/pl_pi_full.json > results/pi/phase_split_latency_real_ffi_pi.json
echo "DONE exit=$?"
