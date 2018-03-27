#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

struct Transform {
	double dx, dy, da;
};

char* videoPath = "./videoplayback2.mp4";
cv::Mat prevImg, curImg;
std::vector<cv::Point2f> prevPoints, curPoints;
std::vector<Transform> transforms;

void init(cv::VideoCapture& videoCapture) {
	videoCapture.open(videoPath);
	if (!videoCapture.isOpened()) {
		std::cerr << "problème lors de la lecture de la video" << std::endl;
	}
	videoCapture >> prevImg;
	cvtColor(prevImg, prevImg, CV_BGR2GRAY);
	cv::goodFeaturesToTrack(prevImg, prevPoints, 1000, 0.01, 10);
}

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status) {
	std::vector<cv::Point2f> result;
	for (int i = 0; i < points.size(); ++i) {
		if (status[i]>0)
			result.push_back(points[i]);
	}
	return result;
}

void precomputeVideo(cv::VideoCapture& videoCapture) {
	int i = 1;
	int j = videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
	do
	{
		std::cout << "compute image " << i << "/" << j << std::endl;
		videoCapture >> curImg;
		if (curImg.empty()) {
			break;
		}
		cvtColor(curImg, curImg, CV_BGR2GRAY);
		std::vector<uchar> status;
		std::vector<float> err;
		cv::goodFeaturesToTrack(curImg, curPoints, 1000, 0.01, 10);
		cv::calcOpticalFlowPyrLK(prevImg, curImg, prevPoints, curPoints, status, err);
		prevPoints = purgePoints(prevPoints, status);
		curPoints = purgePoints(curPoints, status);
		cv::Mat rigidTransform = cv::estimateRigidTransform(prevPoints, curPoints, false);
		if (!rigidTransform.empty()) {
			double dx = rigidTransform.at<double>(0, 2);
			double dy = rigidTransform.at<double>(1, 2);
			double da = atan2(rigidTransform.at<double>(1, 0), rigidTransform.at<double>(0, 0));
			if (!transforms.empty()) {
				dx += transforms.back().dx;
				dy += transforms.back().dy;
				da += transforms.back().da;
			}
			transforms.push_back({ dx, dy, da });
		}
		else {
			transforms.push_back(transforms.back());
			std::cerr << "Pas de transform trouvee" << std::endl;
		}
		prevImg = curImg;
		i++;
	} while (!curImg.empty());
}


int main() {
	cv::namedWindow("input", CV_WINDOW_NORMAL);
	cv::resizeWindow("input", 800, 600);
	cv::VideoCapture videoCapture;
	init(videoCapture);
	precomputeVideo(videoCapture);

	return EXIT_SUCCESS;
}




