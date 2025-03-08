#!/bin/bash

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Force reinstall Cython (fix for cython_sources error)
pip install --no-cache-dir --force-reinstall cython

# Install all dependencies
pip install -r requirements.txt
