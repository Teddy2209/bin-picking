#include<iostream>
#include<stdio.h>
#include<vector>
#include <open3d/Open3D.h>
#include "sl_3D_scanner_api.h"

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iomanip>
#include <fstream>

using namespace std;


bool g_bExit = false;


void Visualize(float *pPointCloud, unsigned char *pColor, int nPoints, const std::string& window_name) {
	
	std::vector<Eigen::Vector3d> pPointcloudSource;
	pPointcloudSource.reserve(nPoints);

	for (int i = 0; i < nPoints; i++)
	{
		pPointcloudSource.push_back(Eigen::Vector3d(pPointCloud[i*3],pPointCloud[i*3+1],pPointCloud[i*3+2]));
	}
	
	open3d::geometry::PointCloud source(pPointcloudSource);

	// why this color doesnt work?
	//for (int i = 0; i < nPoints; i++)
	//{
	//	source.colors_.push_back(Eigen::Vector3d(pColor[i*3],pColor[i*3+1],pColor[i*3+2]));
	//}
	//
	std::shared_ptr<open3d::geometry::PointCloud> source_transformed_ptr(
		new open3d::geometry::PointCloud);

	*source_transformed_ptr = source;

	open3d::visualization::DrawGeometries({source_transformed_ptr}, window_name);
}
void TestVisualize() {
	
	open3d::geometry::PointCloud source;
	open3d::io::ReadPointCloudFromPLY("Save Scan Data/test_.ply", source, open3d::io::ReadPointCloudOption());

	std::shared_ptr<open3d::geometry::PointCloud> source_transformed_ptr(
		new open3d::geometry::PointCloud);

	*source_transformed_ptr = source;

	open3d::visualization::DrawGeometries({source_transformed_ptr});
}

class ScannerCoreTest: public IScannerAPIInterface2Parent
{
public:
	ScannerCoreTest()
	{
		m_pScanCanner = NULL;
	};
	~ScannerCoreTest()
	{

	};

	void ScannerConnect()
	{
		m_pScanCanner = new CSL3DScannerAPI(this);

		// Init projector resolution
		m_pScanCanner->setProjectorResolution(848, 480);

		// Init grabber params
		int nGrabberCount = 2;

		std::vector<CGrabberParam> vecGrabbersParam;

		CGrabberParam grabberParam;
		grabberParam.m_nGrabberLibrary = GrabberLibrary_HIKVISION;
		grabberParam.m_nGrabberInterface = GrabberInterface_Usb;

		grabberParam.m_nFrameWidth = 1280;
		grabberParam.m_nFrameWidthStep = 1024;
		grabberParam.m_nFrameHeight = 1024;
		grabberParam.m_nFrameDepth = 8;
		grabberParam.m_nFrameChannels = 1;

		grabberParam.m_nGrabFrameCount = 1;
		grabberParam.m_nTotalFrameCount = 1;
		grabberParam.m_nTotalScanCount = 1;

		for (size_t nGrabberIdx = 0; nGrabberIdx < nGrabberCount; nGrabberIdx++)
		{
			if (nGrabberIdx == 0)
			{
				grabberParam.m_strConnectInfo = "00E98432421";
			}
			else
			{
				grabberParam.m_strConnectInfo = "00E98432400";
			}

			grabberParam.m_nChannelIndex = nGrabberIdx;
			grabberParam.m_nTotalCameraIndex = nGrabberIdx;

			vecGrabbersParam.push_back(grabberParam);
		}
		//
		// 
		int nRet = m_pScanCanner->ConnectScanner(vecGrabbersParam);

		if (nRet == SUCCESS_RETURN)
		{
			SetSaveScanImages(false);
			SetSave3DOutput(false);
			SetCamExposureTime();
			SetCamTrigger();
			SetProjectionDelay();
			printf("Scanner connected\n");
		}
		
		//
	};

	void ScannerDisconnect()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
	{
		m_pScanCanner->DisconnectScanner();
	};

	void SetSaveScanImages(bool bSave)
	{
		m_pScanCanner->SetSaveCamImages(bSave);
	}
	void SetSave3DOutput(bool bSave)
	{
		m_pScanCanner->SetSave3DPlyOutput(bSave);
	}


	void SetCamExposureTime()
	{   // Minumum ExposureTime = 50 microsecond
		m_pScanCanner->SetExposureTime(0, 1500);
		printf("Current Exposure Cam 0: %f\n", m_pScanCanner->GetExposureTime(0)); 
		m_pScanCanner->SetExposureTime(1, 1500);
		printf("Current Exposure Cam 1: %f\n", m_pScanCanner->GetExposureTime(1)); 
	}

	void SetProjectionDelay()
	{
		m_pScanCanner->setProjectionDelay(96, 0);
	}

	void SetCamTrigger()
	{
		m_pScanCanner->SetExternalTrigger(true, TriggerSource_Hardware_);
	}

	void DoScan()
	{
		m_pScanCanner->Scan3D();
	}
	bool	Scan3DByLoadImages()
	{
		return m_pScanCanner->Scan3D_byLoadImages();
	}
	void ResetScanStatus()
	{
		m_pScanCanner->ResetScanStatus();
	}

public:
	virtual void ISCAPI2P_ExposureFinish(){
		printf("ISC2P_ExposureFinish\n");
	};
	
	virtual bool FindCheckerboard(const cv::Mat &img, cv::Mat &outImg,
						cv::Size patternSize = cv::Size(9,8))
	{
		std::vector<cv::Point2f> corners;
		bool found = cv::findChessboardCorners(img, patternSize, corners,
					cv::CALIB_CB_ADAPTIVE_THRESH |
					cv::CALIB_CB_FAST_CHECK |
					cv::CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			cv::Mat gray;
			if (img.channels() == 3)
				cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			else
				gray = img;

			cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

			img.copyTo(outImg);
			cv::drawChessboardCorners(outImg, patternSize, corners, true);
		}
		else
		{
			img.copyTo(outImg);
		}

		return found;
	}

	bool FindCheckerboardAdvanced(const cv::Mat &imgGray, cv::Mat &outImg,
                              cv::Size patternSize = cv::Size(9,8))
	{
		if (imgGray.empty()) return false;

		std::vector<cv::Point2f> corners;

		// Tìm checkerboard
		bool found = cv::findChessboardCorners(imgGray, patternSize, corners,
											cv::CALIB_CB_ADAPTIVE_THRESH |
											cv::CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			// Nâng cao độ chính xác
			cv::cornerSubPix(imgGray, corners, cv::Size(11,11), cv::Size(-1,-1),
				cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

			// Copy ảnh gốc sang outImg
			imgGray.copyTo(outImg);
			cv::cvtColor(outImg, outImg, cv::COLOR_GRAY2BGR); // để vẽ màu

			// Vẽ checkerboard
			cv::drawChessboardCorners(outImg, patternSize, corners, found);

			// Vẽ số thứ tự corner
			for (size_t i = 0; i < corners.size(); i++)
			{
				cv::Point pt = cv::Point(cvRound(corners[i].x), cvRound(corners[i].y));
				cv::putText(outImg, std::to_string(i), pt, cv::FONT_HERSHEY_SIMPLEX,
							0.5, cv::Scalar(0,0,255), 1, cv::LINE_AA);
			}

			// Lưu tọa độ corner ra file
			std::ofstream fout("corners.txt");
			if (fout.is_open())
			{
				fout << std::fixed << std::setprecision(10);
				fout << "Total corners: " << corners.size() << "\n\n";
				for (size_t i = 0; i < corners.size(); i++)
					fout << "Corner[" << i << "] = " << corners[i].x << ", " << corners[i].y << "\n";
				fout.close();
				std::cout << "Corners saved to corners.txt\n";
			}

			// cv::imshow("Checkerboard", outImg);
			// cv::waitKey(0);
		}
		else
		{
			imgGray.copyTo(outImg);
			std::cout << "Checkerboard NOT detected!\n";
		}

		return found;
	}

	virtual void ISCAPI2P_ScanFinish_checkerboard(int nResult) {
		printf("ISC2P_ScanFinish %d\n", nResult);

		unsigned char *pImage = nullptr;
		int nWidth, nHeight, nPoints;
		float *pPointCloud = nullptr;
		int *pUV = nullptr;
		int *pProfileIndex = nullptr;
		unsigned char *pRGB = nullptr;
		bool b2CamUV;

		m_pScanCanner->GetScanData(&pImage, nWidth, nHeight, &pPointCloud,
								&pUV, &pProfileIndex, &pRGB,
								nPoints, b2CamUV);

		// --- Create windows once ---
		static bool windowsCreated = false;
		if (!windowsCreated) {
			cv::namedWindow("L", cv::WINDOW_NORMAL);
			cv::namedWindow("R", cv::WINDOW_NORMAL);
			cv::namedWindow("L_CB", cv::WINDOW_NORMAL);
			cv::namedWindow("R_CB", cv::WINDOW_NORMAL);
			windowsCreated = true;
		}

		// --- Split cameras ---
		cv::Mat pImgL(cv::Size(nWidth, nHeight), CV_8UC1, pImage);
		cv::Mat pImgR(cv::Size(nWidth, nHeight), CV_8UC1, pImage + nWidth * nHeight);

		cv::imshow("L", pImgL);
		cv::imshow("R", pImgR);
		
		// ===========================================
		// Checkerboard detection
		// ===========================================
		cv::Mat L_CB, R_CB;
		bool foundL = FindCheckerboardAdvanced(pImgL, L_CB);
		bool foundR = FindCheckerboardAdvanced(pImgR, R_CB);

		if (foundL) printf("[L] Checkerboard FOUND\n");
		else        printf("[L] Checkerboard NOT found\n");

		if (foundR) printf("[R] Checkerboard FOUND\n");
		else        printf("[R] Checkerboard NOT found\n");

		if (foundL && foundR) {
			cv::imwrite("/home/apicoo-ai/Music/Linux_ScanAPI_test_20251117/irovi3D_3D_scanner_api/Calibration/left.png", pImgL);
			cv::imwrite("/home/apicoo-ai/Music/Linux_ScanAPI_test_20251117/irovi3D_3D_scanner_api/Calibration/right.png", pImgR);
		}

		cv::imshow("L_CB", L_CB);
		cv::imshow("R_CB", R_CB);

		cv::waitKey(100); // non-blocking

		// Cleanup
		delete [] pPointCloud;
		delete [] pUV;
		delete [] pProfileIndex;
		delete [] pRGB;

		printf("closed 3D viewer\n");
	}

	virtual void ISCAPI2P_ScanFinish_save(int nResult) {
		printf("ISC2P_ScanFinish %d\n", nResult);
		int nVisualizeMode = 0;

		if (nVisualizeMode == 0)
		{
			//	 load online
			unsigned char *pImage;
			int nWidth, nHeight, nPoints;
			float *pPointCloud;
			int *pUV;
			int *pProfileIndex;
			unsigned char *pRGB;
			bool b2CamUV;
			pImage = NULL;
			pPointCloud = NULL;
			pUV = NULL;
			pProfileIndex = NULL;
			pRGB = NULL;

			m_pScanCanner->GetScanData(&pImage, nWidth, nHeight, &pPointCloud, &pUV, &pProfileIndex, &pRGB, nPoints, b2CamUV);
			
			//
			cv::Mat pImgL = cv::Mat(cv::Size(nWidth,nHeight), CV_8UC1, pImage, cv::Mat::AUTO_STEP);
			cv::imshow("L", pImgL);
			cv::imwrite("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/image2d.png", pImgL);
			cv::waitKey();
			cv::destroyWindow("L");

			//
			Visualize(pPointCloud, pRGB, nPoints, "test viewer");

			//
			if ((pPointCloud) != NULL)
			{
				delete [] (pPointCloud);
				(pPointCloud) = NULL;
			}
			if ((pUV) != NULL)
			{
				delete [] (pUV);
				(pUV) = NULL;
			}
			if ((pProfileIndex) != NULL)
			{
				delete [] (pProfileIndex);
				(pProfileIndex) = NULL;
			}
			if ((pRGB) != NULL)
			{
				delete [] (pRGB);
				(pRGB) = NULL;
			}
		}
		else
		{
			// load offline
			TestVisualize();	
		}
		

		printf("closed 3D viewer\n");
	};

	virtual void ISCAPI2P_ScanFinish(int nResult) {
		printf("DEBUG: ISCAPI2P_ScanFinish_save CALLED with nResult=%d\n", nResult);
		int nVisualizeMode = 0;

		if (nVisualizeMode == 0)
		{
			// -----------------------------
			// Load scan data
			// -----------------------------
			unsigned char *pImage = nullptr;
			int nWidth, nHeight, nPoints;
			float *pPointCloud = nullptr;
			int *pUV = nullptr;
			int *pProfileIndex = nullptr;
			unsigned char *pRGB = nullptr;
			bool b2CamUV=true;

			m_pScanCanner->GetScanData(&pImage, nWidth, nHeight, &pPointCloud, &pUV, &pProfileIndex, &pRGB, nPoints, b2CamUV);

			// -----------------------------
			// Create windows (once)
			// -----------------------------
			static bool windowsCreated = false;
			if (!windowsCreated) {
				cv::namedWindow("L", cv::WINDOW_NORMAL);
				cv::namedWindow("R", cv::WINDOW_NORMAL);
				windowsCreated = true;
			}

			// Split left/right images
			cv::Mat pImgL(cv::Size(nWidth, nHeight), CV_8UC1, pImage);
			cv::Mat pImgR(cv::Size(nWidth, nHeight), CV_8UC1, pImage + nWidth * nHeight);

			cv::imshow("L", pImgL);
			cv::imshow("R", pImgR);
			cv::waitKey(); // minimal lag

			// Save images
			cv::imwrite("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/left.png", pImgL);
			cv::imwrite("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/right.png", pImgR);

			cv::Mat imgGray = pImgL;
			cv::Mat imgColor;
			cv::cvtColor(imgGray, imgColor, cv::COLOR_GRAY2BGR);

			// Detect edges
			cv::Mat edges;
			cv::Canny(imgGray, edges, 50, 150);
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
			cv::dilate(edges, edges, kernel);

			// Find contours
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			std::vector<cv::Point> smallBase, largeBase;

			for (auto& c : contours) {
				double area = cv::contourArea(c);
				if (area < 500) continue;

				std::vector<cv::Point> poly;
				cv::approxPolyDP(c, poly, 5.0, true);

				if (poly.size() == 4) {
					if (smallBase.empty())
						smallBase = poly;
					else if (area > cv::contourArea(smallBase)) {
						largeBase = poly;
					} else {
						largeBase = smallBase;
						smallBase = poly;
					}
				}
			}

			// Draw corners for visualization
			auto drawCorners = [&](const std::vector<cv::Point>& pts, const cv::Scalar& color) {
				for (int i = 0; i < pts.size(); i++) {
					cv::circle(imgColor, pts[i], 6, color, -1);
					cv::putText(imgColor, std::to_string(i),
								pts[i] + cv::Point(5,-5),
								cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
				}
			for (int i = 0; i < pts.size(); i++) {
					cv::line(imgColor, pts[i], pts[(i+1)%pts.size()], color, 2);
				}
			};

			if (!smallBase.empty())  drawCorners(smallBase, cv::Scalar(255, 0, 0));
			if (!largeBase.empty())  drawCorners(largeBase, cv::Scalar(0, 255, 0));

			cv::imwrite("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/left_corners.png", imgColor);

			printf("Saved left_corners.png\n");
			// -----------------------------
			// Save point cloud (PLY)
			// -----------------------------
			std::ofstream ply_file("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/pointcloud.ply");
			ply_file << "ply\nformat ascii 1.0\n";
			ply_file << "element vertex " << nPoints << "\n";
			ply_file << "property float x\nproperty float y\nproperty float z\n";
			ply_file << "end_header\n";
			for (int i = 0; i < nPoints; i++) {
				float x = pPointCloud[i * 3 + 0];
				float y = pPointCloud[i * 3 + 1];
				float z = pPointCloud[i * 3 + 2];
				ply_file << x << " " << y << " " << z << "\n";
			}
			ply_file.close();

			// -----------------------------
			// Save UV mapping
			// -----------------------------
			std::ofstream uv_file("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/uv_cam1.txt");
			for (int i = 0; i < nPoints; i++) {
				int u = pUV[i * 2];
				int v = pUV[i * 2 + 1];
				uv_file << u << " " << v << "\n";
			}
			uv_file.close();

			cv::Mat disparityMap = cv::Mat::zeros(nHeight, nWidth, CV_32F); // float disparity
			if (b2CamUV && pUV) {
				for (int i = 0; i < nPoints; ++i) {
					int u1 = pUV[2 * i];
					int v1 = pUV[2 * i + 1];
					int u2 = pUV[2 * nPoints + 2 * i];      // u in cam2 (right image)
					// int v2 = pUV[2 * nPoints + 2 * i + 1]; // not needed for disparity

					// Validate pixel coordinates
					if (u1 >= 0 && u1 < nWidth && v1 >= 0 && v1 < nHeight) {
						float disp = static_cast<float>(u1 - u2);
						// Optional: only accept non-negative disparity
						if (disp >= 0.0f) {
							disparityMap.at<float>(v1, u1) = disp;
						}
					}
				}
			}

			// Optional: visualize or normalize for saving as 8-bit image
			cv::Mat disp8;
			cv::normalize(disparityMap, disp8, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::imwrite("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/disparity.png", disp8);

			// Optional: save raw float disparity (for accuracy)
			cv::FileStorage fs("/home/apicoo-ai/pmg/Scan3d_API_pmg/irovi3D_3D_scanner_api/output_data/disparity.yml", cv::FileStorage::WRITE);
			fs << "disparity" << disparityMap;
			fs.release();

			printf("Saved disparity.png and disparity.yml\n");
			// -----------------------------
			// Optional: visualize point cloud
			// -----------------------------
			if (pPointCloud) {
				Visualize(pPointCloud, nullptr, nPoints, "test viewer"); // RGB not needed
			}

			// -----------------------------
			// Free memory
			// -----------------------------
			delete [] pPointCloud;
			delete [] pUV;
			delete [] pProfileIndex;
			delete [] pRGB;
		}
		else
		{
			// load offline
			TestVisualize();    
		}

		printf("closed 3D viewer\n");
	};

	virtual void ISCAPI2P_ScanFinish_disparitymap(int nResult) {
		printf("ISC2P_ScanFinish %d\n", nResult);

		// 1. Grab images and point cloud
		unsigned char *pImage = nullptr;
		int nWidth = 0, nHeight = 0, nPoints = 0;
		float *pPointCloud = nullptr;
		int *pUV = nullptr;
		int *pProfileIndex = nullptr;
		unsigned char *pRGB = nullptr;
		bool b2CamUV = false;

		m_pScanCanner->GetScanData(&pImage, nWidth, nHeight,
								&pPointCloud, &pUV, &pProfileIndex, &pRGB,
								nPoints, b2CamUV);

		if (!pImage) {
			printf("No image data available!\n");
			return;
		}

		// 2. Convert to OpenCV Mats
		cv::Mat imgL(nHeight, nWidth, CV_8UC1, pImage);
		cv::Mat imgR(nHeight, nWidth, CV_8UC1, pImage + nWidth * nHeight); // offset to right image

		// 3. Create StereoSGBM matcher
		int minDisparity = 0;
		int numDisparities = 16; // must be divisible by 16
		int blockSize = 15;
		
		// cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(
		// 	minDisparity, numDisparities, blockSize
		// );

		cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(
			numDisparities, blockSize
		);

		cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

		// 4. Compute raw disparity
		cv::Mat left_disp, right_disp;
		left_matcher->compute(imgL, imgR, left_disp);
		right_matcher->compute(imgR, imgL, right_disp);

		// 5. Create WLS filter
		cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter =
			cv::ximgproc::createDisparityWLSFilter(left_matcher);
		wls_filter->setLambda(8000.0);
		wls_filter->setSigmaColor(1.5);

		// 6. Filter disparity
		cv::Mat filtered_disp;
		wls_filter->filter(left_disp, imgL, filtered_disp, right_disp);

		// 7. Convert disparities to displayable format
		cv::Mat raw_disp_vis, filtered_disp_vis;
		cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, 1.0);
		cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, 1.0);

		// 8. Show results
		cv::imshow("Left Image", imgL);
		cv::imshow("Right Image", imgR);
		cv::imshow("Raw Disparity", raw_disp_vis);
		cv::imshow("Filtered Disparity", filtered_disp_vis);

		cv::waitKey();  // minimal wait to avoid blocking

		// 9. Optionally visualize 3D point cloud
		if (pPointCloud && pRGB) {
			Visualize(pPointCloud, pRGB, nPoints, "3D Viewer");
		}

		// 10. Free memory
		if (pPointCloud) { delete[] pPointCloud; pPointCloud = nullptr; }
		if (pUV) { delete[] pUV; pUV = nullptr; }
		if (pProfileIndex) { delete[] pProfileIndex; pProfileIndex = nullptr; }
		if (pRGB) { delete[] pRGB; pRGB = nullptr; }

		printf("Closed 3D viewer & disparity display\n");
	};

	virtual void ISCAPI2P_CamCaptureBuffer(int nCamIdx, int nWidth, int nHeight, unsigned char* pImg) {
		printf("ISC2P_CamCaptureBuffer: cam %d\n", nCamIdx);
	};

	virtual void ISCAPI2P_CamUpdateExposureValues(double dCam1Exposure, double dCam2Exposure) {
		printf("ISC2P_CamUpdateExposureValues: cam1 %f, cam2 %f\n", dCam1Exposure, dCam2Exposure);
	};

private:
	CSL3DScannerAPI *m_pScanCanner;
	
};

ScannerCoreTest *pScannerCoreTestPtr;
// wait for user to input enter to stop grabbing or end the sample program
void PressEnterToExit(void)
{
    printf("PressEnterToExit()\n");

    char c = 0;
    while(c == 0)
    {
        printf("waiting for a char\n");
        cin >> c;

        printf("got char %c \n", c);

		if (c == 's')
		{
			/* do scan */
			pScannerCoreTestPtr->DoScan();

			c = 0;
		}
		if (c == 'l')
		{
			/* do scan */
			pScannerCoreTestPtr->SetSave3DOutput(false);
			pScannerCoreTestPtr->SetSaveScanImages(false);
			//pScannerCoreTestPtr->Scan3DByLoadImages();

			c = 0;
		}
		if (c == 'r')
		{
			/* reset scan status */
			pScannerCoreTestPtr->ResetScanStatus();

			c = 0;
		}
		
		if (c=='d')
		{
			pScannerCoreTestPtr->ScannerDisconnect();
			c ==1;
		}
    	sleep(1);
    }

    g_bExit = true;
};

int main_testvisualize()
{
	printf("test open3d\n");

	TestVisualize();

	return 1;
}


int main()
{

    ScannerCoreTest pScanCore;
	pScannerCoreTestPtr = &pScanCore;

	pScanCore.ScannerConnect();
	
    PressEnterToExit();

	pScanCore.ScannerDisconnect();

    return 1;
}