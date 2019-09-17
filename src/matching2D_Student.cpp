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
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      std::string descriptorType,
                      std::string matcherType,
                      std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF")
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType == "MAT_FLANN")
    {
        // Convert binary descriptors to floating point due to a bug in current OpenCV implementation
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
    }
    else
    {
        std:: cerr << "Unknown matcher " << matcherType << std::endl;
        return;
    }

    // perform matching task
    if (selectorType == "SEL_NN")
    {
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType == "SEL_KNN")
    {
        // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;

        const int k = 2;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // Compute distance ratio and only keep non-ambiguous matches
        const float min_desc_dist_ratio = 0.8F;
        for (const std::vector<cv::DMatch>& match_pair : knn_matches)
        {
            const float distance_ratio = match_pair[0].distance / match_pair[1].distance;
            if (distance_ratio < min_desc_dist_ratio)
            {
                matches.push_back(match_pair[0]);
            }
        }
    }
    else
    {
        std::cout << "Unknown selector " << selectorType << std::endl;
    }

    std::cout << "Found " << matches.size() << " matches" << std::endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::Feature2D> extractor;
    if (descriptorType == "BRISK")
    {
        extractor = cv::BRISK::create();
    }
    else if (descriptorType == "BRIEF")
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType == "ORB")
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType == "FREAK")
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType == "AKAZE")
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType == "SIFT")
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cerr << "Unknown descriptor type " << descriptorType << std::endl;
        return;
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

    // Optionally visualize
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, "Harris Corner Detection Results");
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // Create detector. They all inherit from the cv::Feature2D interface class
    cv::Ptr<cv::Feature2D> detector;

    double t = static_cast<double>(cv::getTickCount());

    if (detectorType == "FAST")
    {
        detector = cv::FastFeatureDetector::create();
    }
    else if (detectorType == "BRISK")
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType == "ORB")
    {
        detector = cv::ORB::create();
    }
    else if (detectorType == "AKAZE")
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType == "SIFT")
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cerr << "Unknown detector type " << detectorType << std::endl;
        return;
    }

    // Compute keypoints
    detector->detect(img, keypoints);

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << detectorType << " detection with n=" << keypoints.size() << " keypoints "
              << "in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Optionally visualize
    if (bVis)
    {
        visualizeKeypoints(img, keypoints, detectorType + " Corner Detection Results");
    }
}
