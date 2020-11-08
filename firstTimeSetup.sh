#!/bin/bash

rm -rf venv
python -m venv venv

source venv/bin/activate
python -m pip install pip wheel setuptools --upgrade
pip install -e .