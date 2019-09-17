#include <numeric>
#include "matching2D.h"

namespace
{

void visualizeKeypoints(const cv::Mat& img,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const std::string& window_name)
{
    cv::Mat vis_image = img.clone();
    cv::drawKeypoints(img, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(window_name, vis_image);
    cv::waitKey(0);
}

}  // namespace

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" <<std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int block_size = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * block_size;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = static_cast<double>(cv::getTickCount());
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), block_size, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint new_keypoint;
        new_keypoint.pt = cv::Point2f((*it).x, (*it).y);
        new_keypoint.size = block_size;
        keypoints.push_back(new_keypoint);
    }
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results");
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    const int block_size = 2;     // for every pixel, a block_size × block_size neighborhood is considered
    const int aperture_size = 3;  // aperture parameter for Sobel operator (must be odd)
    const int min_response = 100; // minimum value for a corner in the 8bit scaled response matrix
    const double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners. dst will contain a dense response map
    double t = static_cast<double>(cv::getTickCount());

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);

    // Normalize the response map to lie in the range 0-255
    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Create list of corners by taking the local maxima
    keypoints.clear();
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > min_response)
            {
                cv::KeyPoint new_keypoint;
                new_keypoint.pt = cv::Point2f(i, j);
                new_keypoint.size = 2 * aperture_size;
                new_keypoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool overlap = false;
                for (cv::KeyPoint& keypoint : keypoints)
                {
                    double keypoint_overlap = cv::KeyPoint::overlap(new_keypoint, keypoint);
                    if (keypoint_overlap > maxOverlap)
                    {
                        overlap = true;
                        if (new_keypoint.response > keypoint.response)
                        {
                            // If the new keypoint overlaps with the previous and
                            // the response is larger, replace the old one with the new one
                            keypoint = new_keypoint;
                            break;
                        }
                    }
                }
                // If no overlap has been found for previous keypoints, add to the list
                if (!overlap)
                {
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize keypoints
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, "Harris Corner Detection Results");
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{

}
