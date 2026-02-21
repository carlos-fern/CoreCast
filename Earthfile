VERSION 0.8

ARG ROS_DISTRO=jazzy
ARG UBUNTU_VERSION=24.04

devcontainer-image:
    FROM ubuntu:24.04
    ENV DEBIAN_FRONTEND=noninteractive

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
        libeigen3-dev \
        libyaml-cpp-dev \
        && rm -rf /var/lib/apt/lists/*

    RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
    RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" > /etc/apt/sources.list.d/ros2.list

    RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-dev-tools \
        && rm -rf /var/lib/apt/lists/*

    RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc
    WORKDIR /workspaces/CoreCast

    SAVE IMAGE corecast/dev:jazzy-24.04
