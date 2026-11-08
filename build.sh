#!/bin/sh
set -e

CFLAGS="$(pkg-config --cflags egl openxr wayland-client mpv)"
LIBS="$(pkg-config --libs egl openxr wayland-client mpv) -lGL"

g++ -std=c++20 -O2 -Wall main.cpp -o mpv-vr $CFLAGS $LIBS
