#!/bin/bash
# create environment
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
#pip install -r requirements.txt
# download fasttext
#curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz
#curl -O https://raw.githubusercontent.com/tensorflow/hub/master/examples/text_embeddings_v2/export_v2.py
#gunzip -qf cc.bn.300.vec.gz --k
#python export_v2.py --embedding_file=cc.bn.300.vec --export_path=text_module --num_lines_to_ignore=1 --num_lines_to_use=100000
# run
python __main__.py
deactivate