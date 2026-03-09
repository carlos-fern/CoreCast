# CoreCast

![CoreCast Logo](logo.png)

CoreCast aims to be a high-performance library and ROS 2 framework designed to bridge the gap between NVIDIA's OptiX ray-tracing engine and robotic point cloud processing. By leveraging hardware-accelerated NVIDIA RT Cores, CoreCast enables real-time geometric queries on massive point sets that are traditionally computationally expensive for CPUs.

## Build

```bash
cmake -S corecast_optix -B build/corecast_optix \
  -DOPTIX_SDK_PATH="$PWD/third_party/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/corecast_optix -j
```

This configures and builds the `corecast_optix` library and also writes a CMake package config into `build/corecast_optix`, so other CMake projects can import it with `find_package(corecast_optix CONFIG REQUIRED)`.

## Build ROS 2 package

```bash
colcon build --base-paths corecast_ros2
```

`corecast_ros2` links against `corecast_optix::corecast_optix` through `find_package(corecast_optix CONFIG REQUIRED)`.

## Demo

[![CoreCast Depth Map POC](https://drive.google.com/thumbnail?id=1eFX_67Zpg9qzzgYJ0ya5WZzA7KFfElFe)](https://drive.google.com/file/d/1eFX_67Zpg9qzzgYJ0ya5WZzA7KFfElFe/view?usp=sharing)

## Run

Launch the depth node, point cloud simulator, and Foxglove bridge together:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch corecast_ros2 corecast_ros2_with_sim.launch.py
```

The depth node subscribes to `/point_cloud`, processes each frame through OptiX, and publishes the result as a `32FC1` image on `/depth_map`.

To view in Foxglove Studio, connect to `ws://localhost:8765`.
