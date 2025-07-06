#!/bin/bash

# Check which system dependencies are already installed
# This script checks all packages needed for the locate-3d-preprocessing environment

echo "Checking system dependencies..."
echo "================================"

# List of required packages
packages=(
    "build-essential"
    "cmake"
    "pkg-config"
    "ffmpeg"
    "libopencv-dev"
    "libhdf5-dev"
    "libjpeg-dev"
    "libpng-dev"
    "libtiff-dev"
    "libwebp-dev"
    "libopenjp2-7-dev"
    "libavcodec-dev"
    "libavformat-dev"
    "libswscale-dev"
    "libavresample-dev"
    "libgstreamer1.0-dev"
    "libgstreamer-plugins-base1.0-dev"
    "libgtk-3-dev"
    "libqt6-dev"
    "qt6-base-dev"
    "libssl-dev"
    "libcurl4-openssl-dev"
    "zlib1g-dev"
    "libbz2-dev"
    "liblzma-dev"
    "libxml2-dev"
    "libxslt1-dev"
    "libffi-dev"
    "libsqlite3-dev"
    "libedit-dev"
    "libncurses5-dev"
    "libreadline-dev"
    "tk-dev"
    "libgdbm-dev"
    "libdb-dev"
    "libpcap-dev"
    "xz-utils"
    "curl"
    "llvm"
    "libncursesw5-dev"
    "libgdbm-compat-dev"
    "libc6-dev"
    "openssl"
)

installed=()
missing=()

echo "Checking $(echo ${packages[@]} | wc -w) packages..."
echo ""

for package in "${packages[@]}"; do
    if dpkg -l | grep -q "^ii  $package "; then
        echo "✓ $package (installed)"
        installed+=("$package")
    else
        echo "✗ $package (missing)"
        missing+=("$package")
    fi
done

echo ""
echo "Summary:"
echo "========="
echo "Installed: ${#installed[@]} packages"
echo "Missing: ${#missing[@]} packages"

if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    echo "To install missing packages, run:"
    echo "sudo apt update && sudo apt install -y \\"
    for package in "${missing[@]}"; do
        if [ "$package" == "${missing[-1]}" ]; then
            echo "    $package"
        else
            echo "    $package \\"
        fi
    done
fi

echo ""
echo "Note: Some packages might be installed under different names."
echo "Run 'apt search package-name' to find alternatives." 