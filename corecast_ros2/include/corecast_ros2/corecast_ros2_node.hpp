#ifndef CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_
#define CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_

#include <rclcpp/rclcpp.hpp>

namespace corecast_ros2
{

class CorecastRos2Node : public rclcpp::Node
{
public:
  CorecastRos2Node();
};

}  // namespace corecast_ros2

#endif  // CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_
