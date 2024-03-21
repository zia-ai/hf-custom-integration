#!/usr/bin/env bash
set -ex

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $CUR_DIR


python3 -m poetry run ./main.py "$@"

