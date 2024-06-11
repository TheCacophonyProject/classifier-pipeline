#!/bin/bash
set -xe
python3 -m build -w -C="--global-option=--plat-name" -C="--global-option=manylinux_2_28_aarch64.whl"
python3 -m twine upload  dist/*
