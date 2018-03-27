#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

#include <cstdlib>


struct Transform {
    float dx;
    float dy;
    float da;
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
    }
    else {
        std::cout << "premiere image" << std::endl;
        lastFrame = img;
        detectPoints(lastFrame);
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

    float borderWidth = roiBorder * width;
    float borderHeight = roiBorder * height;
    roi = cv::Rect(borderWidth, borderHeight, width - 2 * borderWidth, height - 2 * borderHeight);

    cv::Mat nextInput;
    cap >> nextInput;
    int i = 0;
    while (!nextInput.empty()) {

        trackPoints(nextInput);

        draw(nextInput);
        bool read = cap.read(nextInput);
        i++;
        if (cv::waitKey(10) >= 0) break;
    }
}


int main(int argc, char** argv) {
    cv::namedWindow("input", CV_WINDOW_NORMAL);
    cv::resizeWindow("input", 800,600);

    video(videoPath);

    return EXIT_SUCCESS;
}