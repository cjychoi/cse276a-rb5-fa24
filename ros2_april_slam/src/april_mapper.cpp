#include <memory>
#include <math.h>
#include <chrono>
#include "ros2_april_slam/april_slam.h"
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_storage/filesystem_helper.hpp>
#include <rosbag2_storage_default_plugins/sqlite/sqlite_storage.hpp>
#include <rosbag2_test_common/memory_management.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"


// using std::placeholders::_1;
using namespace std;

// mapping
AprilSlam l_map;
Vector3 odom = {0.0, 0.0, 0.0};
Vector2 velocities = {0.0, 0.0};
int optimizer_trigger = 0, imu_trigger = 0;

bool imu_init;
double t_prev;

void saveStates(gtsam::Values states, string target_dir){
  return;
}

//callbacks
void aprilCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg){
  
  ++optimizer_trigger;
  double x = msg->pose.position.x;
  double y = msg->pose.position.z;

  auto t_start = std::chrono::high_resolution_clock::now();
  l_map.updateMeasurement(odom, Vector2(x, y), stoi(msg->header.frame_id));
  auto t_stop = std::chrono::high_resolution_clock::now();
  odom[0] = odom[1] = odom[2] = 0.0;
  auto update_dt = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start);

  if (optimizer_trigger % 20 == 0 ){
    t_start = std::chrono::high_resolution_clock::now();
    gtsam::Values res_states = l_map.optimizeGraph();
    int graph_size = res_states.size();

    t_stop = std::chrono::high_resolution_clock::now();
    update_dt = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start);
    std::cout << "Optimization dt: " << update_dt.count() << " | ";
    std::cout << "Graph size: " << graph_size << std::endl;
    optimizer_trigger = 0;
  }

  return;
}

void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg){

  ++imu_trigger;
  if (!imu_init) {
    imu_init = true;
    t_prev = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    return;
  }
  
  
  double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  double dt = t - t_prev;
  t_prev = t;

  double w_z = msg->angular_velocity.z;
  double a_x = msg->linear_acceleration.x;
  double a_y = msg->linear_acceleration.y;

  velocities[0] += a_x * dt;
  velocities[1] += a_y * dt;

  odom[0] +=  velocities[0] * dt; // x = x(0) + a_x * dt^2
  odom[1] +=  velocities[1] * dt; // y = y(0) + a_y * dt^2
  odom[2] +=  w_z * dt; // yaw

  return;
}

std::vector<std::shared_ptr<rosbag2_storage::SerializedBagMessage>> getMessages(std::string bag_path)
{
  std::vector<std::shared_ptr<rosbag2_storage::SerializedBagMessage>> table_msgs;
  auto storage = std::make_shared<rosbag2_storage_plugins::SqliteStorage>();
  storage->open(bag_path, rosbag2_storage::storage_interfaces::IOFlag::READ_ONLY);

  while (storage->has_next()) {
    table_msgs.push_back(storage->read_next());
  }

  return table_msgs;
}

template<typename MessageT>
std::vector<std::shared_ptr<MessageT>> processTopic(std::vector<std::shared_ptr<rosbag2_storage::SerializedBagMessage>> bag, std::string topic_name){
  rosbag2_test_common::MemoryManagement memory_management;
  auto topic_messages = std::vector<std::shared_ptr<MessageT>>();

  for (const auto &msg : bag){
    if( msg->topic_name == topic_name){
      auto msg_ros = memory_management.deserialize_message<MessageT>(msg->serialized_data);
      topic_messages.push_back(msg_ros);
    }
  }
  
  return topic_messages;
}

void processImuCamera(std::vector<std::shared_ptr<rosbag2_storage::SerializedBagMessage>> bag){

  // bagfile processing
  rosbag2_test_common::MemoryManagement memory_management;
  auto imu_messages = std::vector<std::shared_ptr<sensor_msgs::msg::Imu>>();
  auto april_messages = std::vector<std::shared_ptr<geometry_msgs::msg::PoseStamped>>();

  // extract imu and april detections separately
  for (const auto &msg : bag){
    if( msg->topic_name == "/imu"){
      auto msg_ros = memory_management.deserialize_message<sensor_msgs::msg::Imu>(msg->serialized_data);
      imu_messages.push_back(msg_ros);
      std::cout << std::fixed;
    }
    else if( msg->topic_name == "/april_poses"){
      auto msg_ros = memory_management.deserialize_message<geometry_msgs::msg::PoseStamped>(msg->serialized_data);
      april_messages.push_back(msg_ros);
    }
  }

  std::cout << "Processing IMU msgs: " << imu_messages.size() << " | April Detections: " << april_messages.size() << std::endl;

  // process in order
  double imu_ts, det_ts;
  while (imu_messages.size() && april_messages.size()){

    imu_ts = imu_messages[0]->header.stamp.sec + imu_messages[0]->header.stamp.nanosec * 1e-9;
    det_ts = april_messages[0]->header.stamp.sec + april_messages[0]->header.stamp.nanosec * 1e-9;
    if (imu_ts < det_ts){
      // process imu
      imuCallback(imu_messages[0]);
      imu_messages.erase(imu_messages.begin());
    }
    else{
      // process detection
      aprilCallback(april_messages[0]);
      april_messages.erase(april_messages.begin());

    }
  }
  //process remaining imu
  while (imu_messages.size()){
    imu_ts = imu_messages[0]->header.stamp.sec + imu_messages[0]->header.stamp.nanosec * 1e-9;
    det_ts = april_messages[0]->header.stamp.sec + april_messages[0]->header.stamp.nanosec * 1e-9;
    imuCallback(imu_messages[0]);
    imu_messages.erase(imu_messages.begin());

  }
  // process remaining april detections
  while (april_messages.size()){
    imu_ts = imu_messages[0]->header.stamp.sec + imu_messages[0]->header.stamp.nanosec * 1e-9;
    det_ts = april_messages[0]->header.stamp.sec + april_messages[0]->header.stamp.nanosec * 1e-9;
    aprilCallback(april_messages[0]);
    april_messages.erase(april_messages.begin());

  }

  // save states
  // Values final_states = l_map.getStates(); 

  // get landmarks
  auto landmarks = l_map.getLandmarks(); 
  for (auto landmark: landmarks){
    cout << "matrix: " << landmark(0) << " | " << landmark(1) << endl;
  }
}
int main(int argc, char* argv[]){

  std::string dir_path = "/home/dpaz/Documents/gtsam_data/rosbag2_2022_02_09-02_25_26";

  auto bag = getMessages(dir_path);
  processImuCamera(bag);


  return 0;
}