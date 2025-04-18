#!/bin/bash
set -e
pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install -r requirements.txt
module load ffmpeg
module load libtool
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
export OPENBLAS_NUM_THREADS=4  # or 8, 16 etc - a reasonable number for your system
export OMP_NUM_THREADS=4
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa