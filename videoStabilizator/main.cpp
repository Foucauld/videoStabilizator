#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

struct Transform {
	double dx, dy, da;

    Transform operator+(const Transform& t) {
        return {dx + t.dx, dy + t.dy, da + t.da};
    }

    Transform operator-(const Transform& t) {
        return {dx - t.dx, dy - t.dy, da - t.da};
    }

    Transform operator*(double f) {
        return {dx * f, dy * f, da * f};
    }
};

typedef Transform Trajectory;

char* videoPath = "./cut.mp4";
cv::Mat prevImg, curImg;
std::vector<cv::Point2f> prevPoints, curPoints;
std::vector<Transform> transforms;
std::vector<Trajectory> trajectories;
std::vector<Trajectory> smoothedTrajectories;

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
            transforms.push_back({dx, dy, da});
			if (!trajectories.empty()) {
				dx += trajectories.back().dx;
				dy += trajectories.back().dy;
				da += trajectories.back().da;
			}
			trajectories.push_back({ dx, dy, da });
		}
		else {
			trajectories.push_back(trajectories.back());
			std::cerr << "Pas de transform trouvee" << std::endl;
		}
		prevImg = curImg;
		i++;
	} while (!curImg.empty());
}

void computeSmoothTrajectories() {

    int chunkSize = 30;
    int accCtr = 0;


    for (int i = 0; i < trajectories.size(); ++i) {
        Trajectory trj{0, 0, 0};
        for (int j = i - chunkSize; j < i + chunkSize; ++j) {

            if (j >= 0) {

                trj = trj + trajectories[j];
                ++accCtr;
            }
        }

        smoothedTrajectories.push_back(trj * (1.0f / (float)accCtr));
        accCtr = 0;
    }
}

void applyTransforms(cv::VideoCapture& videoCapture) {

    videoCapture.set(CV_CAP_PROP_POS_FRAMES, 0);

    cv::Mat currentFrame;
    for (int i = 0; i < transforms.size(); ++i) {

        videoCapture >> currentFrame;

        Transform currentTransform = smoothedTrajectories[i];
        Trajectory lastTrajectory{0, 0, 0};

        if (i != 0) {
            lastTrajectory = trajectories[i - 1];
        }

        Transform imageTransform{currentTransform.dx - lastTrajectory.dx,
                                 currentTransform.dy - lastTrajectory.dy,
                                 currentTransform.da - lastTrajectory.da};

        cv::Mat affineTrans(2, 3, CV_64FC1);

        affineTrans.at<double>(0, 0) = cos(imageTransform.da);
        affineTrans.at<double>(0, 1) = -sin(imageTransform.da);
        affineTrans.at<double>(1, 0) = sin(imageTransform.da);
        affineTrans.at<double>(1, 1) = cos(imageTransform.da);
        affineTrans.at<double>(0, 2) = imageTransform.dx;
        affineTrans.at<double>(1, 2) = imageTransform.dy;

        cv::warpAffine(currentFrame, currentFrame, affineTrans, currentFrame.size());
        cv::imshow("input", currentFrame);
        cv::waitKey(10);
    }
}

int main() {
	cv::namedWindow("input", CV_WINDOW_NORMAL);
	cv::resizeWindow("input", 800, 600);
	cv::VideoCapture videoCapture;
	init(videoCapture);
	precomputeVideo(videoCapture);
    computeSmoothTrajectories();
    applyTransforms(videoCapture);

	return EXIT_SUCCESS;
}




