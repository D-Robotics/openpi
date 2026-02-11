#!/bin/bash

os_type=$(uname)
binary="OpenExplorerDocServer"

if [ "$os_type" = "Darwin" ]; then
    arch=$(uname -m)
    if [ "$arch" = "x86_64" ]; then
        echo "Current arch is x86_64"
        binary="${binary}_darwin_amd64_v1"
    elif [ "$arch" = "arm64" ]; then
        echo "Current arch is arm64"
        binary="${binary}_darwin_arm64"
    else
        echo "Unsupported arch: $arch"
        exit 1
    fi
fi

chmod +x "./$binary"
"./$binary"
