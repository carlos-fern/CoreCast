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

## Run Example

```bash
build/corecast_optix/corecast_optix_pointcloud_hello_world
```

The point cloud example writes output artifacts to the build directory.
