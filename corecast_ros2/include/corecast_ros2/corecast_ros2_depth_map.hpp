#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "corecast_processing/corecast_depth_map.hpp"

namespace corecast::ros2
{

class CoreCastRos2DepthMap
{
public:
    explicit CoreCastRos2DepthMap(const std::string& node_name)
    : node_(std::make_shared<rclcpp::Node>(node_name))
    {}

    ~CoreCastRos2DepthMap() = default;

    void process_depthmap(const sensor_msgs::msg::PointCloud2& point_cloud)
    {
      (void)point_cloud;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscription_;
    std::unique_ptr<corecast::processing::CoreCastDepthMap<corecast::optix::PointXYZI>> depth_map_processor_;
};

}  // namespace corecast::ros2