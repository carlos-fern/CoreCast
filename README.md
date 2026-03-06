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

## Run Example

```bash
build/corecast_optix/corecast_optix_hello_world
```

The example writes an output image to `corecast_output.ppm`.
