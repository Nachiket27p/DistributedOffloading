#!/bin/bash
sudo add-apt-repository ppa:deadsnakes -y
sudo apt update -y
sudo apt install git -y
sudo apt install net-tools -y
sudo apt install python3.8 -y
sudo apt install python3-pip -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 2
sudo pip install numpy
sudo pip install matplotlib
sudo pip install zfpy
