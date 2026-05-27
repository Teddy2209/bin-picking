#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <vector>
#include "sl_3D_scanner_api.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

class ScannerNode : public rclcpp::Node, public IScannerAPIInterface2Parent
{
public:
    ScannerNode() : Node("scanner_node"), m_pScanCanner(nullptr)
    {
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/scanner/image_left", 10);
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/scanner/pointcloud", 10);
        uv_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/scanner/uv_mapping", 10);
        trigger_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/scanner/start_scan",
            std::bind(&ScannerNode::handle_scan_request, this, std::placeholders::_1, std::placeholders::_2)
        );
        connect_scanner();
    }

    ~ScannerNode() {
        if (m_pScanCanner) {
            m_pScanCanner->DisconnectScanner();
            delete m_pScanCanner;
        }
    }

private:
    CSL3DScannerAPI *m_pScanCanner;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr uv_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_srv_;
    bool is_scanning_ = false;

    void connect_scanner() {
        m_pScanCanner = new CSL3DScannerAPI(this);
        m_pScanCanner->setProjectorResolution(848, 480);
        std::vector<CGrabberParam> vecGrabbersParam;
        for (int i = 0; i < 2; i++) {
            CGrabberParam gp;
            gp.m_nGrabberLibrary = GrabberLibrary_HIKVISION;
            gp.m_nGrabberInterface = GrabberInterface_Usb;
            gp.m_nFrameWidth = 1280; gp.m_nFrameHeight = 1024;
            gp.m_strConnectInfo = (i == 0) ? "00E98432421" : "00E98432400";
            gp.m_nChannelIndex = i; gp.m_nTotalCameraIndex = i;
            vecGrabbersParam.push_back(gp);
        }
        if (m_pScanCanner->ConnectScanner(vecGrabbersParam) == SUCCESS_RETURN) {
            m_pScanCanner->SetExposureTime(0, 1500);
            m_pScanCanner->SetExposureTime(1, 1500);
            m_pScanCanner->SetExternalTrigger(true, TriggerSource_Hardware_);
            RCLCPP_INFO(this->get_logger(), "Scanner connected");
        }
    }

    void handle_scan_request(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        (void)req;
        if (is_scanning_) { res->success = false; return; }
        is_scanning_ = true;
        m_pScanCanner->Scan3D();
        res->success = true;
    }

public:
    // Đúng các hàm ảo có trong IScannerAPIInterface2Parent
    virtual void ISCAPI2P_ExposureFinish() override { RCLCPP_INFO(this->get_logger(), "Exposure finished"); }
    virtual void ISCAPI2P_CamCaptureBuffer(int nIdx, int nW, int nH, unsigned char* pImg) override { (void)nIdx; (void)nW; (void)nH; (void)pImg; }
    virtual void ISCAPI2P_CamUpdateExposureValues(double d1, double d2) override { (void)d1; (void)d2; }

    virtual void ISCAPI2P_ScanFinish(int nResult) override {
        RCLCPP_INFO(this->get_logger(), "Scan finished: %d", nResult);
        unsigned char *pImage = nullptr; float *pPointCloud = nullptr;
        int nW, nH, nPoints, *pUV = nullptr, *pIdx = nullptr; unsigned char *pRGB = nullptr;
        bool b2CamUV = true; // Cần biến thực cho tham chiếu bool&
        m_pScanCanner->GetScanData(&pImage, nW, nH, &pPointCloud, &pUV, &pIdx, &pRGB, nPoints, b2CamUV);
        if (pImage && pPointCloud) {
            std_msgs::msg::Header header; header.stamp = this->now(); header.frame_id = "camera_frame";
            cv::Mat pImgL(cv::Size(nW, nH), CV_8UC1, pImage);
            image_pub_->publish(*(cv_bridge::CvImage(header, "mono8", pImgL).toImageMsg()));
            sensor_msgs::msg::PointCloud2 pc_msg;
            pc_msg.header = header; pc_msg.height = 1; pc_msg.width = nPoints;
            sensor_msgs::PointCloud2Modifier modifier(pc_msg);
            modifier.setPointCloud2FieldsByString(1, "xyz"); modifier.resize(nPoints);
            sensor_msgs::PointCloud2Iterator<float> it_x(pc_msg, "x"), it_y(pc_msg, "y"), it_z(pc_msg, "z");
            for (int i = 0; i < nPoints; i++, ++it_x, ++it_y, ++it_z) {
                *it_x = pPointCloud[i*3]; *it_y = pPointCloud[i*3+1]; *it_z = pPointCloud[i*3+2];
            }
            pc_pub_->publish(pc_msg);
        }
        if (pPointCloud) delete[] pPointCloud; if (pUV) delete[] pUV;
        if (pIdx) delete[] pIdx; if (pRGB) delete[] pRGB;
        is_scanning_ = false;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScannerNode>());
    rclcpp::shutdown();
    return 0;
}
