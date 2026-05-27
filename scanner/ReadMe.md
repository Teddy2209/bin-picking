**************************************************************
* This document is generated for IROVI3D scanner and its API *
* Contact: irovi.scanner@gmail.com                           *
**************************************************************

1. Prerequisite
1.1. Hardware and settings
- Scanner includes a camera head and a projector (provided by IROVI3D)
- Camera head is pre-calibrated, and parameters are stored in folder "/Setting/Input/parameter"

1.2. Camera driver instalation
- Cameras are HIKROBOT brand
- link: https://www.hikrobotics.com/en/machinevision/service/download?module=0
- file name: MVS_STD_GML_V2.1.2_221208.zip

1.3 USB port setting - Make rules to access USB

Make file (in any folder) lightcrafter.rules
# for libusb, kernel v < 2.6.24
SUBSYSTEM=="usb_device", ACTION=="add", ATTRS{idVendor}=="0451", ATTRS{idProduct}=="6401", GROUP="lam", MODE="0666"

# for libusb, kernel v > 2.6.24
SUBSYSTEM=="usb", ACTION=="add", ATTRS{idVendor}=="0451", ATTRS{idProduct}=="6401", GROUP="lam", MODE="0666"

# for hidraw version of hidapi
KERNEL=="hidraw*", ATTRS{busnum}=="1", ATTRS{idVendor}=="0451", ATTRS{idProduct}=="6401", GROUP="lam", MODE="0666"

#copy file to "/etc/udev/rules.d/"
sudo cp lightcrafter.rules /etc/udev/rules.d/lightcrafter.rules

1.3. Open3D for 3D viewer test
- Install latest version of open3D would be good


2. Using scanner and API
2.1. Connect scanner
step 1: Assign camera serial numbers and connect the scanner: ConnectScanner()
step 3: Set options:
       + Set camera exposure time (in micro seconds): depending on the object's surface brightness: SetExposureTime()
       + Set save 3D output option (PLY file), the saved file: "SaveScanData/test_.ply"

2.2. Scan function
- After scanner is connected, call "Scan3D()" to start scan
- Sometimes, due to the connection problem / or CPU too busy, the camera cannot capture enough image, the process will be stopped, function "ResetScanStatus()" need to called before do scan again

2.3. Callbacks from scanner API
- Callbacks are managed by interface class "IScannerAPIInterface2Parent"
- Callback functions are:
    + ISCAPI2P_ExposureFinish():            this signal indicates the camera finished capturing image, and start to run 3D reconstruction process
    + ISCAPI2P_ScanFinish():                this signal indicates the 3D reconstruction process is finished, ready to get 3D data from API
    + ISCAPI2P_CamCaptureBuffer():          user can get camera image buffer from this callback function
    + ISCAPI2P_CamUpdateExposureValues():   if camera exposure value is changed, the update value getting from camera is updated to this callback function

2.4. Data format
- Output scan data can get from function: GetScanData(unsigned char **pImage, int &nWidth, int &nHeight, float **pPointCloud, int **pUV, int **pProfileIndex, unsigned char **pRGB, int &nPoints, bool &b2CamUV)
- Parameters: 
    + unsigned char **pImage: pointer of the buffer for the camera 2D image (when projector illuminates white pattern) to be copied to, the buffer contains two images (cam1, cam2) [img_cam1_width_height][img_cam2_width_height]
    + int &nWidth, int &nHeight: size of the camera image
    + float **pPointCloud: pointer of the pointcloud (xyz) to be copied to
    + int **pUV: pUV contains the corresponding image uv value of the 3D point i_th; example:
        u1 = pUV[i*2]        // u in image cam1
        v1 = pUV[i*2+1]      // v in image cam1
        u2 = pUV[nPoints*2+i*2]        // u in image cam2
        v2 = pUV[nPoints*2+i*2+1]      // v in image cam2
    + int **pProfileIndex: the corresponding code value of projector pattern of the 3D point i_th
    + unsigned char **pRGB: the color intensity value (RGB) of the 3D point i_th
    + int &nPoints: total number of 3D point
    + bool &b2CamUV: indicates that buffers contains two images (cam1 cam2) or only cam1
