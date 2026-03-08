#include "corecast_ros2/corecast_ros2_node.hpp"

#include <memory>

namespace corecast_ros2
{

CorecastRos2Node::CorecastRos2Node()
: Node("corecast_ros2_node")
{
  RCLCPP_INFO(this->get_logger(), "corecast_ros2_node started");
}

}  // namespace corecast_ros2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<corecast_ros2::CorecastRos2Node>());
  rclcpp::shutdown();
  return 0;
}
