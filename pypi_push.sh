#!/bin/bash
set -xe
python3 -m build
python3 -m twine upload  dist/*
