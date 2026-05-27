#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <vector>


#ifndef _DEF_SCANNER_ENUMS
#define _DEF_SCANNER_ENUMS
enum SL3DSCANNER_RETURNS {SUCCESS_RETURN = 0, CANNOT_CONNECT_PROJECTOR, CANNOT_CONNECT_CAMERAS, LICENSE_FAIL};
#endif

#ifndef _DEF_GRABBER_PARAMS
#define _DEF_GRABBER_PARAMS

enum GrabberLibraryType		{ GrabberLibrary_Simulation=0, GrabberLibrary_Sapera, GrabberLibrary_Mil, GrabberLibrary_PixeLink, GrabberLibrary_Pylon, GrabberLibrary_Sentech,GrabberLibrary_JAI, GrabberLibrary_HIKVISION, GrabberLibrary_Count };
enum GrabberInterfaceType	{ GrabberInterface_Simulation=0, GrabberInterface_Usb, GrabberInterface_GigE, GrabberInterface_1394, GrabberInterface_CameraLink, GrabberInterface_Count };
enum TriggerModeType		{ TriggerMode_Internal=0, TriggerMode_External, TriggerMode_Count };
enum TriggerSourceType		{ TriggerSource_Software_=0, TriggerSource_Hardware_, TriggerSource_Count }; // hardware ==> Line0
enum GrabFrameType			{ GrabFrame_Circular=0 };
enum ConnectReturnValue		{ ConnectReturn_Fail=-1, ConnectReturn_Success=0 };
enum FrameFlipMode			{ FrameFlip_None = 0, FrameFlip_Vertical, FrameFlip_Horizontal, FrameFlip_Both, FrameFlip_Count };

class CGrabberParam
{
public:
	CGrabberParam()				{ Reset(); }
	virtual ~CGrabberParam()	{ Reset(); }
	void Reset()
	{
		m_nGrabberLibrary	= GrabberLibrary_Simulation;
		m_nGrabberInterface	= GrabberInterface_Simulation;
		m_nGrabberIndex		= 0;
		m_nChannelIndex		= 0;
		m_nTotalCameraIndex	= 0;

		m_nFrameFlipMode	= FrameFlip_None;
		m_nFrameWidth		= 0;
		m_nFrameHeight		= 0;
		m_nFrameWidthStep	= 0;
		m_nFrameDepth		= 8;
		m_nFrameChannels	= 0;
		m_nGrabFrameCount	= 0;



		m_nTotalScanCount	= 1;
		m_nTotalFrameCount	= 0;
		m_strConnectInfo	= "";
	}

public:
	int		m_nGrabberLibrary;		// grabber library
	int		m_nGrabberInterface;	// grabber interface
	int		m_nGrabberIndex;		// grabber index
	int		m_nChannelIndex;		// channel index
	int		m_nTotalCameraIndex;	// total camera index

	int		m_nFrameFlipMode;		// frame flip mode
	int		m_nFrameWidth;			// frame width
	int		m_nFrameHeight;			// frame height
	int		m_nFrameWidthStep;		// frame width step
	int		m_nFrameDepth;			// frame depth
	int		m_nFrameChannels;		// frame channels
	int		m_nGrabFrameCount;		// grab frame count

	int		m_nTotalScanCount;		// total scan count
	int		m_nTotalFrameCount;		// total frame count

	std::string	m_strConnectInfo;		// connect info
};
#endif


#ifndef _DEF_CLASS_IScannerInterface2Parent
#define _DEF_CLASS_IScannerInterface2Parent

class IScannerInterface2Parent
{
	public:
	virtual void ISC2P_ExposureFinish() = 0;
	virtual void ISC2P_ScanFinish(int nResult) = 0;
	virtual void ISC2P_CamCaptureBuffer(int nCamIdx, int nWidth, int nHeight, unsigned char* pImg) = 0;
	virtual void ISC2P_CamUpdateExposureValues(double dCam1Exposure, double dCam2Exposure) = 0;
};

#endif

class IScannerAPIInterface2Parent
{
	public:
	virtual void ISCAPI2P_ExposureFinish() = 0;
	// virtual bool FindCheckerboard(const cv::Mat &img, cv::Mat &outImg, cv::Size patternSize = cv::Size(9,6));
	virtual void ISCAPI2P_ScanFinish(int nResult) = 0;
	//virtual void ISCAPI2P_ScanFinish_save(int nResult) = 0;
	// virtual void ISCAPI2P_ScanFinish_checkerboard(int nResult) = 0;
	// virtual void ISCAPI2P_ScanFinish_disparitymap(int nResult) = 0;

	virtual void ISCAPI2P_CamCaptureBuffer(int nCamIdx, int nWidth, int nHeight, unsigned char* pImg) = 0;
	virtual void ISCAPI2P_CamUpdateExposureValues(double dCam1Exposure, double dCam2Exposure) = 0;
};



class CSL3DScanner;

class CSL3DScannerAPI : public IScannerInterface2Parent
{
public:
    
public:
	CSL3DScannerAPI(IScannerAPIInterface2Parent *pSC2PI);
	~CSL3DScannerAPI(void);

public: // PROCAP interface

	virtual void ISC2P_ExposureFinish();
	virtual void ISC2P_ScanFinish(int nResult);
	virtual void ISC2P_CamCaptureBuffer(int nCamIdx, int nWidth, int nHeight, unsigned char* pImg);
	virtual void ISC2P_CamUpdateExposureValues(double dCam1Exposure, double dCam2Exposure);

public:	// scanner operations

	int		ConnectScanner(std::vector<CGrabberParam> vectorGrabberParam);
	int		DisconnectScanner();
	bool	isScannerConnected();

	void 	setProjectionDelay(int nPreProjectionDelay, int nPostProjectionDelay);
	void 	setProjectorResolution(int nWidth, int nHeight);
	bool	SetExposureTime(int nCamIdx,float fExposure_us);
	bool	SetCameraGain(int nCamIdx,float fGain);
	bool	SetCameraGamma(int nCamIdx,float fGamma);
	bool	SetExternalTrigger(bool bTriggerOnOff, TriggerSourceType nTriggerSource);
	double	GetExposureTime(int nCamIdx);
	void	camSendSWTrigger();

	void 	ResetScanStatus();
	bool	Scan3D();
	int		GetScanData(unsigned char **pImage, int &nWidth, int &nHeight, float **pPointCloud, int **pUV, int **pProfileIndex, unsigned char **pRGB, int &nPoints, bool &b2CamUV);
	bool	Scan3D_byLoadImages();
	 
	void 	SetSaveCamImages(bool bSave);
	void 	SetSave3DPlyOutput(bool bSave);

	bool	GetProcessingState();
    bool	GetExposureFinishState();
	
    // calibration paramters include 7 params:
    // fx,fy, distortion_0~5, cx,cy, width, height
	void 	GetCalibrationParams(float *pCam1Calib_12params, float *pCam2Calib_12params);


private:
	CSL3DScanner	*m_pScanner;
    IScannerAPIInterface2Parent *m_pInterface;
};