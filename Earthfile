VERSION 0.8

devcontainer-image:
    ARG ROS_DISTRO=jazzy
    FROM ros:${ROS_DISTRO}-ros-base-noble
    ENV DEBIAN_FRONTEND=noninteractive
    ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

    RUN curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i /tmp/cuda-keyring.deb \
        && rm -f /tmp/cuda-keyring.deb

    RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg2 \
        lsb-release \
        software-properties-common \
        git \
        build-essential \
        cmake \
        ninja-build \
        python3-pip \
        cuda-toolkit \
        libgl1-mesa-dev \
        libopengl-dev \
        libglx-dev \
        mesa-common-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libx11-dev \
        libeigen3-dev \
        libyaml-cpp-dev \
        && rm -rf /var/lib/apt/lists/*

    RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-dev-tools \
        && rm -rf /var/lib/apt/lists/*

    RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc
    WORKDIR /workspaces/CoreCast

    SAVE IMAGE corecast/dev:jazzy-24.04
