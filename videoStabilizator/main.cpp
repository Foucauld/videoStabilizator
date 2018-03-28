#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <clara.hpp>

using namespace clara;

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

cv::Mat prevImg, curImg;
std::vector<cv::Point2f> prevPoints, curPoints;
std::vector<Transform> transforms;
std::vector<Trajectory> trajectories;
std::vector<Trajectory> smoothedTrajectories;
cv::VideoWriter videoWriter;

void init(cv::VideoCapture& videoCapture, std::string& videoPath, int startFrame) {
	videoCapture.open(videoPath);
	if (!videoCapture.isOpened()) {
		std::cerr << "problème lors de la lecture de la video" << std::endl;
	}

    videoCapture.set(CV_CAP_PROP_POS_FRAMES, startFrame);

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

void precomputeVideo(cv::VideoCapture& videoCapture, int endFrame) {
	int i = 1;
	int j = (int)videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
    int currentFrame;

    if (endFrame <= 0) {
        endFrame = (int)videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
    }

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
        currentFrame = (int)videoCapture.get(CV_CAP_PROP_POS_FRAMES);
	} while (!curImg.empty() && currentFrame < endFrame);
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

void applyTransforms(cv::VideoCapture& videoCapture, bool shouldWriteOutputVideo, bool shouldPreview) {

    int fps = (int)videoCapture.get(CV_CAP_PROP_FPS);
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

        if (shouldPreview) {
            cv::imshow("input", currentFrame);
        }

        if (shouldWriteOutputVideo) {
            videoWriter << currentFrame;
        }

        cv::waitKey(1000 / fps);
    }
}

bool initVideoWriter(const std::string& filename, const cv::Size& size, int fps) {
    videoWriter.open(filename, CV_FOURCC('W', 'M', 'V', '2'), fps, size, true);
    return videoWriter.isOpened();
}

int main(int argc, const char * const *argv) {

    std::string inputVideo = "";
    std::string outputVideo = "";
    int startFrame = 0;
    int endFrame = 0;
    bool shouldWriteOutputVideo = false;
    bool shouldPreview = false;
    auto cli =
            Opt(inputVideo, "inputVideo")
                ["-i"]["--input"]
                ("Path to the input video to stabilize") |
            Opt(outputVideo, "outputVideo")
                ["-o"]["--output"]
                ("Path to the output video") |
            Opt(shouldPreview)
                ["-p"]["--preview"]
                ("Should we display a video preview") |
            Opt(startFrame, "startFrame")
                ["-s"]["--start-frame"]
                ("Start from frame") |
            Opt(endFrame, "endFrame")
                ["-e"]["--end-frame"]
                ("End at frame");

    auto result = cli.parse(Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line : " << result.errorMessage() << std::endl;
        return EXIT_FAILURE;
    }

    if (inputVideo.empty()) {
        std::cerr << "Missing parameter inputVideo (--input)" << std::endl;
        return EXIT_FAILURE;
    }

    shouldWriteOutputVideo = !outputVideo.empty();

    if (shouldPreview) {
        cv::namedWindow("input", CV_WINDOW_NORMAL);
        cv::resizeWindow("input", 800, 600);
    }

	cv::VideoCapture videoCapture;
    init(videoCapture, inputVideo, startFrame);

    cv::Size videoSize((int)videoCapture.get(CV_CAP_PROP_FRAME_WIDTH), (int)videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT));
    int fps = (int)videoCapture.get(CV_CAP_PROP_FPS);

	precomputeVideo(videoCapture, endFrame);
    computeSmoothTrajectories();

    if (shouldWriteOutputVideo) {
        bool videoWriterOk = initVideoWriter(outputVideo, videoSize, fps);
        std::cout << "Video writer is ok : " << videoWriterOk << std::endl;
    }

    applyTransforms(videoCapture, shouldWriteOutputVideo, shouldPreview);

    videoWriter.release();

	return EXIT_SUCCESS;
}




