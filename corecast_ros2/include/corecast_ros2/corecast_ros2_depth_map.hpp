#pragma once

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>

#include <cstring>
#include <functional>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <string>
#include <vector>

#include "corecast_processing/corecast_depth_map.hpp"

namespace corecast::ros2 {

class CoreCastRos2DepthMap {
 public:
  explicit CoreCastRos2DepthMap(rclcpp::Node& node)
      : node_(node),
        options_(make_optix_options()),
        optix_(/*context_id=*/0, options_),
        tf_buffer_(node_.get_clock()),
        tf_listener_(tf_buffer_),
        camera_frame_(node_.declare_parameter<std::string>("camera_frame", "camera_link")) {
    tf_buffer_.setCreateTimerInterface(
        std::make_shared<tf2_ros::CreateTimerROS>(node_.get_node_base_interface(), node_.get_node_timers_interface()));
    point_cloud_subscription_ = node_.create_subscription<sensor_msgs::msg::PointCloud2>(
        "point_cloud", 10, std::bind(&CoreCastRos2DepthMap::process_depthmap, this, std::placeholders::_1));
    depth_image_publisher_ = node_.create_publisher<sensor_msgs::msg::Image>("depth_map", rclcpp::QoS(10));
  }

  ~CoreCastRos2DepthMap() = default;

  void process_depthmap(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!msg) {
      return;
    }

    point_cloud_.clear();
    point_cloud_.reserve(static_cast<size_t>(msg->width) * msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> x_it(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_it(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_it(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> intensity_it(*msg, "intensity");

    for (; x_it != x_it.end(); ++x_it, ++y_it, ++z_it, ++intensity_it) {
      corecast::optix::PointXYZI p{};
      p.x = *x_it;
      p.y = *y_it;
      p.z = *z_it;
      p.intensity = *intensity_it;
      p.ring = 0;
      p.timestampOffset = 0.0;
      point_cloud_.push_back(p);
    }

    if (point_cloud_.empty()) {
      return;
    }
    corecast::optix::CameraFrameData camera_frame_data{};
    fill_camera_frame_data(*msg, camera_frame_data);

    depth_map_processor_ = std::make_unique<corecast::processing::CoreCastDepthMap<corecast::optix::PointXYZI>>(
        optix_, point_cloud_, msg->width, msg->height, camera_frame_data, 0.01f);

    sensor_msgs::msg::Image depth_msg;
    depth_msg.header = msg->header;
    depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_msg.is_bigendian = false;
    depth_msg.width = msg->width;
    depth_msg.height = msg->height;
    depth_msg.step = msg->width * static_cast<unsigned int>(sizeof(float));
    depth_msg.data.resize(static_cast<size_t>(depth_msg.step) * depth_msg.height);
    std::memcpy(depth_msg.data.data(), depth_map_processor_->launch_depth_map(), depth_msg.data.size());

    depth_image_publisher_->publish(depth_msg);

    point_cloud_.clear();
    depth_map_processor_.reset();
  }

 private:
  bool fill_camera_frame_data(const sensor_msgs::msg::PointCloud2& msg,
                              corecast::optix::CameraFrameData& camera_frame_data) {
    try {
      const auto tf_msg_frame_from_camera = tf_buffer_.lookupTransform(
          msg.header.frame_id, camera_frame_, msg.header.stamp, rclcpp::Duration::from_seconds(0.05));
      const auto& t = tf_msg_frame_from_camera.transform.translation;
      const auto& q = tf_msg_frame_from_camera.transform.rotation;

      camera_frame_data.sensor_origin =
          make_float3(static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z));

      tf2::Quaternion quat(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 rot(quat);
      // Columns are camera basis vectors expressed in the point cloud frame.
      camera_frame_data.sensor_x_axis =
          make_float3(static_cast<float>(rot[0][0]), static_cast<float>(rot[1][0]), static_cast<float>(rot[2][0]));
      camera_frame_data.sensor_y_axis =
          make_float3(static_cast<float>(rot[0][1]), static_cast<float>(rot[1][1]), static_cast<float>(rot[2][1]));
      camera_frame_data.sensor_z_axis =
          make_float3(static_cast<float>(rot[0][2]), static_cast<float>(rot[1][2]), static_cast<float>(rot[2][2]));
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
                           "TF lookup failed for camera_frame='%s' -> cloud_frame='%s': %s", camera_frame_.c_str(),
                           msg.header.frame_id.c_str(), ex.what());
      return false;
    }
  }

  static OptixDeviceContextOptions make_optix_options() {
    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4;
    return options;
  }

  rclcpp::Node& node_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher_;
  OptixDeviceContextOptions options_{};
  corecast::optix::CoreCastOptix optix_;
  std::unique_ptr<corecast::processing::CoreCastDepthMap<corecast::optix::PointXYZI>> depth_map_processor_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string camera_frame_;
  std::vector<corecast::optix::PointXYZI> point_cloud_;
  bool depth_map_initialized_ = false;
  unsigned int depth_map_width_ = 0;
  unsigned int depth_map_height_ = 0;
};

}  // namespace corecast::ros2