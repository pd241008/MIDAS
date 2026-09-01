#!/bin/bash
# On-device SLA/throughput run (#5). Sets env and runs the load-test binary.
# Output json is written to ~/midas/results/pi/sla_load_test.json by the binary.
cd ~/midas
export EDGE_CONFIG=configs/edge_config.json
export EDGE_MODEL_PATH=models/classifier_float32.tflite
export LOAD_N=50000
export LOAD_SCALE=5
export LOAD_CPU_SECS=4
./target/release/sla_load_test results/pi/load_adv_batch.json results/pi/sla_load_test.json
echo "EXIT=$?"
