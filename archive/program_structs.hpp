// Helloworld image types TODO: Remove this in favor of pointcloud types
struct Params {
    uchar4 *data;
    unsigned int image_width;
    unsigned int image_height;
  };
  
  struct RayGenData {
    float r, g, b;
    float circle_center_x;
    float circle_center_y;
    float circle_radius;
  };
  
  // Core pointcloud type
  struct __attribute__((packed)) PointXYZI {
    float x;
    float y;
    float z;
    float intensity;
    uint16_t ring;
    double timestampOffset;
  };
  
  struct PointCloudLaunchParams {
    unsigned int image_width;
    unsigned int image_height;
  
    // Sensor position and orientation basis vectors
    float3 sensor_origin;
    float3 sensor_x_axis;
    float3 sensor_y_axis;
    float3 sensor_z_axis;
  
    // Tracing limits
    float t_min;
    float t_max;
  
    // The 3D scene handle
    OptixTraversableHandle handle;
  
    // Output buffer to store the depth values (1D array mapped to 2D screen)
    float *depth_buffer;
  };
  
  // Global raygen data (The map)
  struct PointCloudRayGenData {
    PointXYZI *data;
    unsigned int num_points;
  };
  
  struct PointCloudHitData {
    float3 *points;
    float3 *colors;
    float radius;
  };
  
  struct PointCloudMissData {
    float empty_color;
  };
  
  struct PointCloudRayPayload {
    PointXYZI point;
  };
  
  struct CameraFrameData {
    float3 sensor_origin;
    float3 sensor_x_axis;
    float3 sensor_y_axis;
    float3 sensor_z_axis;
  };
  