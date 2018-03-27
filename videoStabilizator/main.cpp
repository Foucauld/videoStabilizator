#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>


struct Transform {
    double dx;
    double dy;
    double da;
};

cv::Rect roi;
float roiBorder = 0.1f;
char* videoPath = "./videoplayback2.mp4";
cv::Mat lastFrame;
std::vector<cv::Point2f> lastPoints;
std::vector<Transform> transforms;

void detectPoints(cv::Mat & img) {

    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    cv::Mat mask(img.size(), CV_8UC1);
    // on met les pixels noirs
    mask = (int)255;
    cv::rectangle(mask, roi, cv::Scalar(0, 0, 0), CV_FILLED);

    cv::goodFeaturesToTrack(grayImg, lastPoints, 1000, 0.01, 10, mask);
}

// Dessin des points de prevPoints
void drawCircles(cv::Mat & img) {
    for (int i = 0; i < lastPoints.size(); i++)
    {
        cv::circle(img, lastPoints[i], 4, cv::Scalar(rand(), rand(), rand()));
    }
}

void drawLines(cv::Mat & img, std::vector<cv::Point2f> prevPoints) {
    for (int i = 0; i < prevPoints.size(); i++)
    {
        cv::circle(img, lastPoints[i], 4, cv::Scalar(rand(), rand(), rand()));
        cv::line(img, prevPoints[i], prevPoints[i], cv::Scalar(rand(), rand(), rand()));
    }
}

cv::Mat applyTransformToImage(const Transform& t, const cv::Mat& originalImg) {

    cv::Mat img = originalImg.clone();
    cv::Mat trans_mat(2, 3, CV_64FC1);

    trans_mat.at<double>(0, 0) = 1;
    trans_mat.at<double>(0, 1) = 0;
    trans_mat.at<double>(0, 2) = - t.dx;
    trans_mat.at<double>(1, 0) = 0;
    trans_mat.at<double>(1, 1) = 1;
    trans_mat.at<double>(1, 2) = - t.dy;

    cv::warpAffine(img, img, trans_mat, img.size());
    return img;
}

void draw(cv::Mat nextInput) {
    cv::Mat img = nextInput.clone();
    // dans les prochaines étapes, c’est ici que l’on mettra les choses a dessiner

    drawCircles(img);
    cv::rectangle(img, roi, cv::Scalar(rand(), rand(), rand()));
    cv::imshow("input", img);
}

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status) {
    std::vector<cv::Point2f> result;
    for (int i = 0; i < points.size(); ++i) {
        if (status[i]>0)
            result.push_back(points[i]);
    }
    return result;
}

void trackPoints(cv::Mat & img) {
    //si prevInput n’est pas vide
    if (!lastFrame.empty()) {
//        if (lastPoints.size() <= 20) {
//            detectPoints(lastFrame);
////            // - > si prevPoints ne contient toujours pas assez de points,
////            // quitter la fonction
//            if (lastPoints.size() <= 20){
//                std::cout << "plus de points a track" << std::endl;
//                return;
//            }
//        }
        std::vector<cv::Point2f> prevPoints = lastPoints;
        detectPoints(img);
        // - > calculer a l’aide de calcOpticalFlowPyrLK les nouvelles positions
        // (nextPoints) dans la nouvelle image (nextImage)
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(img, lastFrame, lastPoints, prevPoints, status, err);
        // -> supprimer les points non suivis des deux listes de points
        //prevPoints = purgePoints(prevPoints, status);
        //lastPoints = purgePoints(lastPoints, status);
        //cloner nextInput dans prevInput
        lastFrame = img.clone();
        drawLines(img, prevPoints);
        cv::Mat rigidTransform = cv::estimateRigidTransform(prevPoints, lastPoints, false);

        double dx = rigidTransform.at<double>(0, 2);
        double dy = rigidTransform.at<double>(1, 2);
        double da = atan2(rigidTransform.at<double>(1, 0), rigidTransform.at<double>(0, 0));

        if (!transforms.empty()) {
            dx += transforms.back().dx;
            dy += transforms.back().dy;
            da += transforms.back().da;
        }

        transforms.push_back({dx, dy, da});
    }
    else {
        std::cout << "premiere image" << std::endl;
        lastFrame = img;
        detectPoints(lastFrame);
    }
}

void computeVideo(char* videoname) {
    cv::VideoCapture cap;
    if(videoname!= nullptr){
        cap.open(videoname);
    } else{
        cap.open(0);
    }
    if (!cap.isOpened()) {
        return;
    }

    float width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    float height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cap.get(CV_CAP_PROP_FRAME_COUNT);

    float borderWidth = roiBorder * width;
    float borderHeight = roiBorder * height;
    roi = cv::Rect(borderWidth, borderHeight, width - 2 * borderWidth, height - 2 * borderHeight);

    cv::Mat nextInput;
    cap >> nextInput;
    int i = 0;
    int j = cap.get(CV_CAP_PROP_FRAME_COUNT);
    while (!nextInput.empty()) {
        std::cout << "compute image " << i << "/" << j << std::endl;
        trackPoints(nextInput);

        cap.read(nextInput);
        i++;
        if (cv::waitKey(10) >= 0) break;
    }
}

void video(char* videoname) {
    cv::VideoCapture cap;
    if(videoname!= nullptr){
        cap.open(videoname);
    } else{
        cap.open(0);
    }
    if (!cap.isOpened()) {
        return;
    }

    float width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    float height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cap.get(CV_CAP_PROP_FRAME_COUNT);

    cv::Mat nextInput;
    cap >> nextInput;
    int i =0;
    while (!nextInput.empty()) {

        cap.read(nextInput);
        cv::Mat newFrame = applyTransformToImage(transforms.at(i), nextInput);
        cv::imshow("before", nextInput);
        cv::imshow("after", newFrame);
        i++;
        if (cv::waitKey(10) >= 0) break;
    }
}


int main(int argc, char** argv) {
    cv::namedWindow("input", CV_WINDOW_NORMAL);
    cv::resizeWindow("input", 800,600);
    computeVideo(videoPath);
    video(videoPath);

    return EXIT_SUCCESS;
}