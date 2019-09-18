/* INCLUDES FOR THIS PROJECT */
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.h"

namespace
{
template <typename T>
void displayStatistics(const std::string& name, const std::vector<T>& data)
{
    std::cout << name << *std::min_element(data.begin(), data.end()) << " - "
                      << *std::max_element(data.begin(), data.end()) << std::endl;
}
}  // namespace

void runExperiment(const DetectorType& detectorType, const DescriptorType& descriptorType)
{
    std::cout << "=======================================================" << std::endl;
    std::cout << "Experiment: " << detectorType << " + " << descriptorType << std::endl;
    std::cout << "=======================================================" << std::endl;
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // Experiment configuration
    const DescriptorFormat descriptorFormat = getDescriptorFormat(descriptorType);
    const MatcherType matcherType = MatcherType::BF;  // Requested in rubric
    const SelectorType selectorType = SelectorType::KNN;  // To use distance ratio of 0.8

    // Buffers to hold and compute statistics
    std::vector<double> computation_time_detector;
    std::vector<double> computation_time_descriptor;
    std::vector<std::size_t> detected_keypoints;
    std::vector<std::size_t> matched_keypoints;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        DataFrame frame{};
        frame.cameraImg = imgGray;

        if (dataBuffer.size() < dataBufferSize)  // If the buffer is not yet full, simply push back
        {
            dataBuffer.push_back(frame);
        }
        else  // Otherwise shift data and place new frame in the back
        {
            // Shift contents in ring buffer
            for (std::size_t i = 1U; i < dataBuffer.size(); ++i)
            {
                dataBuffer[i - 1U] = dataBuffer[i];
            }

            // Add new image to the back
            dataBuffer.back() = frame;
        }

        //// EOF STUDENT ASSIGNMENT
        // std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        bool visualize_detections = false;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType == DetectorType::SHITOMASI)
        {
            detKeypointsShiTomasi(keypoints, imgGray, visualize_detections, computation_time_detector);
        }
        else if (detectorType == DetectorType::HARRIS)
        {
            detKeypointsHarris(keypoints, imgGray, visualize_detections, computation_time_detector);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, visualize_detections, computation_time_detector);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        std::vector<cv::KeyPoint> filtered_keypoints;
        if (bFocusOnVehicle)
        {
            for (const cv::KeyPoint& keypoint : keypoints)
            {
                if (vehicleRect.contains(keypoint.pt))
                {
                    filtered_keypoints.push_back(keypoint);
                }
            }
        }

        keypoints = filtered_keypoints;

        std::cout << "Preceeding vehicle keypoints: " << keypoints.size() << std::endl;
        detected_keypoints.push_back(keypoints.size());
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType == DetectorType::SHITOMASI)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!" << std::endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.back().keypoints = keypoints;
        // std::cout << "#2 : DETECT KEYPOINTS done" << std::endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        descKeypoints(dataBuffer.back().keypoints,
                      dataBuffer.back().cameraImg,
                      descriptors,
                      descriptorType,
                      computation_time_descriptor);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        dataBuffer.back().descriptors = descriptors;

        // std::cout << "#3 : EXTRACT DESCRIPTORS done" << std::endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorFormat, matcherType, selectorType);
            matched_keypoints.push_back(matches.size());

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            dataBuffer.back().kptMatches = matches;

            // std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = dataBuffer.back().cameraImg.clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                std::stringstream ss;
                ss << "[" << detectorType << ", " << descriptorType << "] "
                   << "Matching keypoints between two camera images";
                std::string windowName = ss.str();
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                // std::cout << "Press key to continue to next image" << std::endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

        std::cout << "-------------------------------------------------------------" << std::endl;

    } // eof loop over all images

    // Display statistics for Excel
    std::cout << "------------------ STATISTICS--------------";
    displayStatistics("Detected keypoints: ", detected_keypoints);
    displayStatistics("Matched keypoints: ", matched_keypoints);
    displayStatistics("Detector computation time (ms): ", computation_time_detector);
    displayStatistics("Descriptor computation time (ms): ", computation_time_descriptor);

    std::cout << "Total computation time (ms): "
              << (*std::min_element(computation_time_detector.begin(), computation_time_detector.end()) +
                  *std::min_element(computation_time_descriptor.begin(), computation_time_descriptor.end()))
              << " - "
              << (*std::max_element(computation_time_detector.begin(), computation_time_detector.end()) +
                  *std::max_element(computation_time_descriptor.begin(), computation_time_descriptor.end()))
              << std::endl;
}

bool isValidExperiment(const DetectorType& detector_type, const DescriptorType& descriptor_type)
{
    // Cases documented not to work on UdacityHub
    bool output = true;

    if ((descriptor_type == DescriptorType::AKAZE) && (detector_type != DetectorType::AKAZE))
    {
        // AZAKE descriptor can only be used with KAZE or AKAZE keypoints
        output = false;
    }
    else if ((detector_type == DetectorType::SIFT) && (descriptor_type == DescriptorType::ORB))
    {
        // out-of-memory errors with this combination
        output = false;
    }

    return output;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // Define experiments
    const std::vector<DetectorType> detectors =
    {
        DetectorType::AKAZE,
        DetectorType::BRISK,
        DetectorType::FAST,
        DetectorType::HARRIS,
        DetectorType::ORB,
        DetectorType::SHITOMASI,
        DetectorType::SIFT,
    };

    const std::vector<DescriptorType> descriptors =
    {
        DescriptorType::BRISK,
        DescriptorType::BRIEF,
        DescriptorType::ORB,
        DescriptorType::FREAK,
        DescriptorType::AKAZE,
        DescriptorType::SIFT
    };

    // Run all combinations
    for (const DetectorType& detector_type : detectors)
    {
        for (const DescriptorType& descriptor_type : descriptors)
        {
            if (isValidExperiment(detector_type, descriptor_type))
            {
                runExperiment(detector_type, descriptor_type);
            }
        }
    }

    return 0;
}
