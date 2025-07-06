# System Dependencies

Install these system packages before installing Python packages with uv:

## Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libopencv-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libqt6-dev \
    qt6-base-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libsqlite3-dev \
    libedit-dev \
    libncurses5-dev \
    libreadline-dev \
    tk-dev \
    libgdbm-dev \
    libdb-dev \
    libpcap-dev \
    xz-utils \
    curl \
    llvm \
    libncursesw5-dev \
    libgdbm-compat-dev \
    libc6-dev \
    libssl-dev \
    openssl
```

## For CUDA support (if you have NVIDIA GPU):
```bash
# Install NVIDIA drivers and CUDA toolkit
# Follow NVIDIA's official installation guide for your system
# The pip packages will handle the Python CUDA bindings
```

## CentOS/RHEL/Fedora:
```bash
sudo dnf install -y \
    gcc gcc-c++ make cmake \
    pkg-config \
    ffmpeg-devel \
    opencv-devel \
    hdf5-devel \
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    libwebp-devel \
    openjpeg2-devel \
    qt6-qtbase-devel \
    openssl-devel \
    libcurl-devel \
    zlib-devel \
    bzip2-devel \
    xz-devel \
    libxml2-devel \
    libxslt-devel \
    libffi-devel \
    sqlite-devel \
    readline-devel \
    tk-devel \
    gdbm-devel \
    ncurses-devel
``` 