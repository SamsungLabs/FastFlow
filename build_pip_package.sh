#!/bin/bash

# ------- build requirements -------
pip install "protobuf<3.20,>=3.9.2"
pip install "grpcio-tools>=1.24.3,<1.46.0"
# ----------------------------------
python setup.py clean --all
python setup.py bdist_wheel
INSTALL_FILE=`ls dist/ -ltr | tail -1 | awk '{print $9}'`
pip install dist/${INSTALL_FILE} 
