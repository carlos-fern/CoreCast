#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "corecast_processing/corecast_depth_map.hpp"

namespace corecast::ros2
{

class CoreCastRos2DepthMap
{
public:
  explicit CoreCastRos2DepthMap(rclcpp::Node& node)
  : node_(node),
    options_(make_optix_options()),
    optix_(/*context_id=*/0, options_),
    depth_map_processor_(std::make_unique<corecast::processing::CoreCastDepthMap<corecast::optix::PointXYZI>>(optix_))
  {
    point_cloud_subscription_ = node_.create_subscription<sensor_msgs::msg::PointCloud2>(
      "point_cloud", 10, std::bind(&CoreCastRos2DepthMap::process_depthmap, this, std::placeholders::_1));
    depth_image_publisher_ = node_.create_publisher<sensor_msgs::msg::Image>("depth_map", rclcpp::QoS(10));
  }

  ~CoreCastRos2DepthMap() = default;

  void process_depthmap(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
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

    const unsigned int depth_width = msg->width;
    const unsigned int depth_height = msg->height;
    
    depth_map_width_ = depth_width;
    depth_map_height_ = depth_height;
    depth_map_processor_->setup_depth_map_processing(point_cloud_, depth_map_width_, depth_map_height_);

    float* depth_buffer = depth_map_processor_->launch_depth_map();

    sensor_msgs::msg::Image depth_msg;
    depth_msg.header = msg->header;
    depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_msg.is_bigendian = false;
    depth_msg.width = depth_map_width_;
    depth_msg.height = depth_map_height_;
    depth_msg.step = depth_map_width_ * static_cast<unsigned int>(sizeof(float));
    depth_msg.data.resize(static_cast<size_t>(depth_msg.step) * depth_msg.height);
    std::memcpy(depth_msg.data.data(), depth_buffer, depth_msg.data.size());

    depth_image_publisher_->publish(depth_msg);
  }

private:
  static OptixDeviceContextOptions make_optix_options()
  {
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
  std::vector<corecast::optix::PointXYZI> point_cloud_;
  bool depth_map_initialized_ = false;
  unsigned int depth_map_width_ = 0;
  unsigned int depth_map_height_ = 0;
};

}  // namespace corecast::ros2