#!/bin/bash
python train.py 2>&1 | tee train.log
/usr/bin/shutdown