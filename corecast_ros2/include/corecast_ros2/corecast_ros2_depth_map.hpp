#pragma once

#include "corecast_ros2/corecast_ros2_node.hpp"
#include "corecast_optix/corecast_depth_map.hpp"

namespace corecast::ros2
{

class CoreCastRos2DepthMap
{
public:
    CoreCastRos2DepthProcessor(const std::string& node_name);
    ~CoreCastRos2DepthProcessor();

    void process_depthmap(const sensor_msgs::msg::PointCloud2& point_cloud);

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscription_;
    corecast_optix::CoreCastDepthMap<corecast_optix::PointXYZI> depth_map_processor_;
}

}