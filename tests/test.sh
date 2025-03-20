#!/bin/bash

echo "Running tests..."
python3 -m unittest tests/test_generate.py
python3 -m unittest tests/test_simulate.py
python3 -m unittest tests/test_recover.py