#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>

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
            "/spot/lidar/points", 10, std::bind(&LiDARCamCalibration::lidar_callback, this, std::placeholders::_1));

        // // FOR DEBUGGING: Publisher for filtered point cloud
        filtered_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/filtered_points", 10);
        
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

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;

        // Keep points from 0.1m to 4m in front of the sensor (X-axis)
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.1, 4.0);
        pass.filter(*cloud_filtered);

        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-2.0, 2.0);
        pass.filter(*cloud_filtered);

        if (cloud_filtered->points.size() < 10) { // Need a few points to find a plane
            RCLCPP_WARN(this->get_logger(), "Not enough points after filtering.");
            return;
        }

        // pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        // pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        // // Create the segmentation object
        // pcl::SACSegmentation<pcl::PointXYZ> seg;
        // // Optional
        // seg.setOptimizeCoefficients (true);
        // // Mandatory
        // seg.setModelType (pcl::SACMODEL_PLANE);
        // seg.setMethodType (pcl::SAC_RANSAC);
        // seg.setDistanceThreshold (0.01);

        // seg.setInputCloud (cloud);
        // seg.segment (*inliers, *coefficients);

        // if (inliers->indices.size () == 0)
        // {
        //   PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
        //   return (-1);
        // }

        // FOR DEBUGGING: Publish the filtered point cloud
        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*cloud_filtered, filtered_msg);
        filtered_msg.header = msg->header; // Preserve the original header (frame_id and timestamp)
        filtered_cloud_publisher_->publish(filtered_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_publisher_; // Publisher declaration
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