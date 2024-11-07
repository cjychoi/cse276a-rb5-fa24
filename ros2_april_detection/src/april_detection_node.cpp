#include "ros2_april_detection/april_detection.h"
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>

#include <string>
#include <sstream>


using std::placeholders::_1;
// camera parameters
double distortion_coeff[5] = {0.022327, 
                            -0.019742, 
                            -0.000961, 
                            0.000625, 
                            0.000000};

double intrinsics[9] = {691.01615,    0.     ,  954.51,
                      0.     ,  690.10114,  540.77467,
                      0.     ,    0.     ,    1.};

const cv::Mat d(cv::Size(1, 5), CV_64FC1, distortion_coeff);
const cv::Mat K(cv::Size(3, 3), CV_64FC1, intrinsics);
// TODO: Set tagSize for pose estimation, assuming same tag size.
// details from: https://github.com/AprilRobotics/apriltag/wiki/AprilTag-User-Guide#pose-estimation
const double tagSize = 0.162; // in meters

class AprilDetectionNode : public rclcpp::Node{
  public:

    
    AprilDetectionNode() : Node("april_detection"){
      image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera_0", 10, std::bind(&AprilDetectionNode::imageCallback, this, _1));

      // pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/april_poses", 10);
      pose_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/april_poses", 10);

      tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

      AprilDetection det;

    }

    // undistort raw image data
    cv::Mat rectify(const cv::Mat image) {
      cv::Mat image_rect = image.clone();
      const cv::Mat new_K = cv::getOptimalNewCameraMatrix(K, d, image.size(), 1.0); 
      cv::undistort(image, image_rect, K, d, new_K); 

    // set info for pose estimation using new camera matrix
      det.setInfo(tagSize, new_K.at<double>(0,0), new_K.at<double>(1,1), new_K.at<double>(0,2), new_K.at<double>(1,2));
      
      return image_rect;

    }
    

  private:

    // apriltag lib
    AprilDetection det;

    // subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // publishers
    // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub;

    // tf broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;

    void publishTransforms(vector<apriltag_pose_t> poses, vector<int> ids, std_msgs::msg::Header header){
      tf2::Quaternion q;
      tf2::Matrix3x3 so3_mat;
      tf2::Transform tf;

      // geometry_msgs::msg::TransformStamped tf_gm;
      geometry_msgs::msg::PoseArray pose_array_msg;
      pose_array_msg.header = header;

      // ros2_april_detection::msg::AprilTagDetectionArray apriltag_detection_array_msg;
      // apriltag_detection_array_msg.header = header;
      std::ostringstream oss;
      for (unsigned int i=0; i<poses.size(); i++){
        oss << "marker_" << ids[i] << ",";

        tf.setOrigin(tf2::Vector3(poses[i].t->data[0], 0, poses[i].t->data[2]));
        so3_mat.setValue(poses[i].R->data[0],0,-poses[i].R->data[6],
                         0,1,0,
                         poses[i].R->data[6],0,poses[i].R->data[0]);
        
        double roll, pitch, yaw; 

        // orientation - q
        so3_mat.getRPY(roll, pitch, yaw); // so3 to RPY
        q.setRPY(roll, pitch, yaw);

        tf.setRotation(q);

        tf2::Transform tf_inv = tf.inverse();
        tf2::Vector3 tran_inv = tf_inv.getOrigin();
        tf2::Quaternion q_inv = tf_inv.getRotation();

        string marker_name = "marker_" + to_string(ids[i]);
        string camera_name= "camera_" + to_string(ids[i]);

        geometry_msgs::msg::TransformStamped tf_gm;
        tf_gm.header.stamp = this->get_clock()->now();
        tf_gm.header.frame_id = marker_name;
        tf_gm.child_frame_id = camera_name;

        tf_gm.transform.translation.x = tran_inv.getX();
        tf_gm.transform.translation.y = tran_inv.getY();
        tf_gm.transform.translation.z = tran_inv.getZ();

        tf_gm.transform.rotation.x = q_inv.x();
        tf_gm.transform.rotation.y = q_inv.y();
        tf_gm.transform.rotation.z = q_inv.z();
        tf_gm.transform.rotation.w = q_inv.w();
        tf_->sendTransform(tf_gm);
        // RCLCPP_INFO(this->get_logger(), "Sending TF");

        // Prepare PaseArray
        geometry_msgs::msg::Pose pose;
        pose.position.x = poses[i].t->data[0];
        pose.position.y = poses[i].t->data[1];
        pose.position.z = poses[i].t->data[2];
        
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
        pose_array_msg.poses.push_back(pose);

      }
      pose_array_msg.header.frame_id = oss.str();
      pose_pub->publish(pose_array_msg);
      
    }
    // callbacks
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
      
      cv_bridge::CvImagePtr img_cv = cv_bridge::toCvCopy(msg);
      std_msgs::msg::Header header = msg->header;

      // rectify and run detection (pair<vector<apriltag_pose_t>, cv::Mat>)
      auto april_obj =  det.processImage(rectify(img_cv->image));

      publishTransforms(get<0>(april_obj), get<1>(april_obj), header);
      // geometry_msgs::msg::PoseStamped pose_msg;
      // pose_pub_->publish(pose_msg);

      return;
    }

};

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AprilDetectionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;

}
