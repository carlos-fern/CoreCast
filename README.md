# CoreCast

![CoreCast Logo](logo.png)

CoreCast is a project I am using to expand my personal knowledge around ray tracing, CUDA development, and point cloud processing. This project aims to provide a wrapper around the NVIDIA OptiX framework to process point clouds in robotic applications leveraging NVIDIA RT Cores. The goal is both a standalone library and a ROS 2 node.

## Build

```bash
cmake -S corecast_optix -B build/corecast_optix \
  -DOPTIX_SDK_PATH="$PWD/third_party/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/corecast_optix -j
```

## Run Example

```bash
build/corecast_optix/corecast_optix_hello_world
```

The example writes an output image to `corecast_output.ppm`.
