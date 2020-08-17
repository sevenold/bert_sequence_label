#!/usr/bin/env bash
docker run -t --rm -p 8501:8501 \
-v "$(pwd)/saved_model:/models/docker_test" \
-e MODEL_NAME=docker_test tensorflow/serving
