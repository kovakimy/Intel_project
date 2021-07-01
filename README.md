# Object Tracking with Line Crossing and Area Intrusion Detection
This program uses an object detection deep learning model and a re-identification model to find and track the objects in a movie. Then, the program will track the trajectory of the objects and check if the objects cross the defined virtual lines or the objects are inside the defined areas.

In this repository was implemented detection, re-identification, tracking and rendering on video.

### Object Tracking and Line Crossing Demo
![sample](media/demo_example.gif "Sample")


## How to Run


### 0. Prerequisites
- **OpenVINO 2021.3**
  - If you haven't installed it, go to the OpenVINO web page and follow the [*Get Started*](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) guide to do it.  


### 1. Install dependencies  
The demo depends on:
- some dependencies


### 2. Download DL models from OMZ
Use `Model Downloader` to download the required models and convert the downloaded model into OpenVINO IR models with `Model Converter`.  
``` sh
(Linux) python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
(Win10) python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst
```

### 3. Run the demo app

Firstly in derictory `..\IntelSWTools\openvino_2020.3.194\bin` run `setupvars.bat`.

After that:

``` bash
git clone https://github.com/kovakimy/Intel_project.git
cd Intel_project
mkdir build
cd build
cmake ..
cmake —build
AreaInstrusionDetection.exe
```

## Demo Output  
The application draws the results on the screen.
