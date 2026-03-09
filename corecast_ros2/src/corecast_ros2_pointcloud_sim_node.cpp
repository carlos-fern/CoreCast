#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <string>

namespace corecast::ros2 {

class CorecastRos2PointcloudSimNode : public rclcpp::Node {
 public:
  CorecastRos2PointcloudSimNode() : Node("corecast_ros2_pointcloud_sim_node"), cloud_phase_(0.0), frame_phase_(0.0) {
    static_pointcloud_ = declare_parameter<bool>("static_pointcloud", true);
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", rclcpp::QoS(10));
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    timer_ = create_wall_timer(std::chrono::milliseconds(10),
                               std::bind(&CorecastRos2PointcloudSimNode::publish_pointcloud, this));
    RCLCPP_INFO(get_logger(),
                "PointCloud2 simulator publishing to /point_cloud with TF sim_lidar -> camera_link "
                "(static_pointcloud=%s)",
                static_pointcloud_ ? "true" : "false");
  }

 private:
  void publish_sensor_tf(const rclcpp::Time& stamp, float t) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "sim_lidar";
    tf_msg.child_frame_id = "camera_link";

    // Slightly moving camera pose in the simulated lidar frame.
    tf_msg.transform.translation.x = 0.4 * std::sin(t * 0.4f);
    tf_msg.transform.translation.y = 0.3 * std::cos(t * 0.6f);
    tf_msg.transform.translation.z = 0.2 + 0.2 * std::sin(t * 0.5f);

    tf2::Quaternion q;
    q.setRPY(0.1 * std::sin(t * 0.8f), 0.2 * std::cos(t * 0.6f), t * 0.25f);
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf_msg);
  }

  void publish_pointcloud() {
    // Dense spherical shell with strong global deformation for clear motion from any viewpoint.
    constexpr uint32_t kWidth = 720;  // azimuth samples
    constexpr uint32_t kHeight = 96;  // elevation rings
    constexpr float kRadius = 8.0f;
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kPi = 3.14159265358979323846f;

    const float frame_t = static_cast<float>(frame_phase_);
    const float cloud_t = static_pointcloud_ ? 0.0f : static_cast<float>(cloud_phase_);
    const auto stamp = now();
    publish_sensor_tf(stamp, frame_t);

    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "sim_lidar";
    msg.height = kHeight;
    msg.width = kWidth;
    msg.is_dense = true;
    msg.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(
        6, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z", 1,
        sensor_msgs::msg::PointField::FLOAT32, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, "ring", 1,
        sensor_msgs::msg::PointField::UINT16, "time", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(static_cast<size_t>(kWidth) * kHeight);

    sensor_msgs::PointCloud2Iterator<float> x_it(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> y_it(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> z_it(msg, "z");
    sensor_msgs::PointCloud2Iterator<float> intensity_it(msg, "intensity");
    sensor_msgs::PointCloud2Iterator<uint16_t> ring_it(msg, "ring");
    sensor_msgs::PointCloud2Iterator<float> time_it(msg, "time");

    const float yaw = 0.75f * std::sin(cloud_t * 0.7f);
    const float pitch = 0.55f * std::cos(cloud_t * 0.5f);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);

    for (uint32_t ring = 0; ring < kHeight; ++ring) {
      const float v = (static_cast<float>(ring) + 0.5f) / static_cast<float>(kHeight);
      const float elevation = -0.5f * kPi + v * kPi;  // [-pi/2, pi/2]
      const float cos_el = std::cos(elevation);
      const float sin_el = std::sin(elevation);

      for (uint32_t i = 0; i < kWidth; ++i, ++x_it, ++y_it, ++z_it, ++intensity_it, ++ring_it, ++time_it) {
        const float azimuth = static_cast<float>(i) * (kTwoPi / static_cast<float>(kWidth));
        const float wave1 = std::sin(azimuth * 6.0f + cloud_t * 2.2f);
        const float wave2 = std::cos(elevation * 5.0f - cloud_t * 1.7f);
        const float wave3 = std::sin((azimuth + elevation) * 4.0f + cloud_t * 1.3f);
        const float shell_breath = 1.0f + 0.16f * std::sin(cloud_t * 1.1f);
        const float r = kRadius * shell_breath + 1.6f * wave1 + 1.2f * wave2 + 0.8f * wave3;

        // Base point on deformed sphere.
        float x = r * cos_el * std::cos(azimuth);
        float y = r * cos_el * std::sin(azimuth);
        float z = r * sin_el;

        // Add large translational waves so the whole sphere shifts over time.
        x += 1.8f * std::sin(cloud_t * 0.9f);
        y += 1.4f * std::cos(cloud_t * 1.3f);
        z += 1.2f * std::sin(cloud_t * 1.6f);

        // Rotate by pitch then yaw for dramatic global movement.
        const float y_pitch = cp * y - sp * z;
        const float z_pitch = sp * y + cp * z;
        const float x_yaw = cy * x + sy * z_pitch;
        const float z_yaw = -sy * x + cy * z_pitch;

        *x_it = x_yaw;
        *y_it = y_pitch;
        *z_it = z_yaw;
        *intensity_it = 120.0f + 100.0f * std::sin(azimuth * 2.0f + elevation * 3.0f + cloud_t * 2.4f);
        *ring_it = static_cast<uint16_t>(ring);
        *time_it = (static_cast<float>(ring) * static_cast<float>(kWidth) + static_cast<float>(i)) /
                   static_cast<float>(kWidth * kHeight);
      }
    }

    publisher_->publish(msg);
    if (!static_pointcloud_) {
      cloud_phase_ += 0.22;
    }
    frame_phase_ += 0.22;
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool static_pointcloud_;
  double cloud_phase_;
  double frame_phase_;
};

}  // namespace corecast::ros2

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<corecast::ros2::CorecastRos2PointcloudSimNode>());
  rclcpp::shutdown();
  return 0;
}
