#!/bin/bash

set -eux

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get install -y flex neovim tldr bison automake libtool m4 pkg-config libyajl-dev libmicrohttpd-dev libhiredis-dev

if [ ! -d "/usr/include/onnxruntime" ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz
    tar -xvf onnxruntime-linux-aarch64-1.20.1.tgz || tar -xvf onnxruntime-linux-aarch64-1.20.1.tgz
    cd onnxruntime-linux-aarch64-1.20.1

    sudo mv lib/libonnxruntime* /usr/lib
    sudo mkdir -p /usr/include/onnxruntime
    sudo mv include/* /usr/include/onnxruntime

    cat lib/pkgconfig/libonnxruntime.pc | sed 's/lib64/lib/g' | sed s+/local++g > new_pc
    sudo mv new_pc /usr/lib/pkgconfig/libonnxruntime.pc
    cd ..
fi

./build.sh && ./configure && make && sudo make install
sudo ln -sfn collectd /opt/collectd
./collectd -f -C collectd.conf
