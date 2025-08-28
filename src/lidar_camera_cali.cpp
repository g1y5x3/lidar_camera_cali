#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp" // Required for cv::imshow


extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tagStandard41h12.h"
}

class LiDARCamCalibration : public rclcpp::Node
{
public:
    LiDARCamCalibration()
    : Node("lidar_cam_cali_node")
    {
        // Create a subscription to the front-left fisheye camera image topic
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/spot/camera/frontright_fisheye/image_raw", 10, std::bind(&LiDARCamCalibration::image_callback, this, std::placeholders::_1));

        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/spot/lidar/points", 10, std::bind(&AprilTagDetector::lidar_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "Node started. Subscribing to camera and lidar topics.");

        // Initialize AprilTag detector
        tf_ = tagStandard41h12_create();
        td_ = apriltag_detector_create();
        apriltag_detector_add_family(td_, tf_);
    }

    ~LiDARCamCalibration()
    {
        // Clean up the AprilTag detector and family
        apriltag_detector_destroy(td_);
        tagStandard41h12_destroy(tf_);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS Image message to an OpenCV Mat
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // cv::Mat rotated_image;
        // cv::rotate(cv_ptr->image, rotated_image, cv::ROTATE_90_COUNTERCLOCKWISE);
        // cv::imshow("Rotated Camera Image", rotated_image);
        // cv::waitKey(1); // Required to display the image

        // Convert the color image to grayscale
        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        // Prepare the image for the AprilTag detector
        image_u8_t im = {
            .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };

        // Run the detector
        zarray_t *detections = apriltag_detector_detect(td_, &im);

        RCLCPP_INFO(this->get_logger(), "Detected %d tags", zarray_size(detections));

        // Loop through detections and print information
        for (int i = 0; i < zarray_size(detections); i++)
        {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            RCLCPP_INFO(this->get_logger(), "  > Tag ID: %d, Center: (%.2f, %.2f)", det->id, det->c[0], det->c[1]);
        }

        // Free the memory used by the detections
        apriltag_detections_destroy(detections);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    apriltag_family_t *tf_;
    apriltag_detector_t *td_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LiDARCamCalibration>());
    rclcpp::shutdown();
    return 0;
}