#include "corecast_ros2/corecast_ros2_node.hpp"

#include <memory>

namespace corecast::ros2
{

CorecastRos2Node::CorecastRos2Node()
: Node("corecast_ros2_node"),
  depth_map_processor_(*this)
{
  RCLCPP_INFO(this->get_logger(), "corecast_ros2_node started");
}

}  // namespace corecast::ros2

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<corecast::ros2::CorecastRos2Node>());
  rclcpp::shutdown();
  return 0;
}
