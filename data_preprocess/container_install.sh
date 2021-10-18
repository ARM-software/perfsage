/usr/bin/python3 -m pip install --upgrade pip
pip install jupyter 

jupyter notebook --generate-config
cp jupyter_notebook_config.py /root/.jupyter/

pip install tensorflow==2.4.0
pip install networkx
pip install matplotlib
pip install ethos-u-vela
