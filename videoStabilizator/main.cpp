#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\video\video.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void drawCircles(cv::Mat & img);
void drawLines(cv::Mat & img);
void draw();
void detectPoints(cv::Mat & img);
void trackPoints();
std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void video(char* videoname);
void updateROI();

cv::Mat nextInput;
std::vector<cv::Point2f> prevPoints;

cv::Mat prevInput;
std::vector<cv::Point2f> nextPoints;

cv::Rect roi;
cv::Point start(-1, -1);

// Dessin des points de prevPoints
void drawCircles(cv::Mat & img) {
	for (int i = 0; i < prevPoints.size(); i++)
	{
		cv::circle(img, prevPoints[i], 4, cv::Scalar(rand(), rand(), rand()));
	}
}

// Dessin des points de nextpoints avec une ligne correspondante avec ceux de prevPoints
void drawLines(cv::Mat & img) {
	for (int i = 0; i < prevPoints.size(); i++)
	{
		cv::circle(img, nextPoints[i], 4, cv::Scalar(rand(), rand(), rand()));
		cv::line(img, prevPoints[i], nextPoints[i], cv::Scalar(rand(), rand(), rand()));
	}
}

void draw() {
	cv::Mat img = nextInput.clone();
	// dans les prochaines étapes, c’est ici que l’on mettra les choses a dessiner

	//drawCircles(img);
	drawLines(img);
	cv::rectangle(img, roi, cv::Scalar(rand(), rand(), rand()));
	cv::imshow("input", img);
	//si il y a un call back de souris a mettre, ca sera ici
	cv::setMouseCallback("input", CallBackFunc, nullptr);
}

void detectPoints(cv::Mat & img) {
	//utiliser la fonction goodFeaturesToTrack pour detecter les points d’interet
	// dans img et les stocker dans prevPoints
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	cv::Mat mask(img.size(), CV_8UC1);
	// on met les pixels noirs
	mask = (int)0;
	cv::rectangle(mask, roi, cv::Scalar(255, 255, 255), CV_FILLED);

	if (roi.area() < 10 || start.x >= 0) return;

	cv::goodFeaturesToTrack(grayImg, prevPoints, 1000, 0.01, 10, mask);

}

void updateROI() {
	//si roi est vide ou si une selection est en cours, on quitte direct
	if (roi.area() < 10 || start.x >= 0) return;
	//sinon, on calcule le rectangle englobant notre ensemble de points stockés
	// dans nextPoints, et on l’assigne a roi
	roi = cv::boundingRect(nextPoints);
}

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status) {
	std::vector<cv::Point2f> result;
	for (int i = 0; i < points.size(); ++i) {
		if (status[i]>0)result.push_back(points[i]);
	}
	return result;
}

void trackPoints() {
	//si prevInput n’est pas vide
	if (!prevInput.empty()) {
		// - > mettre dans prevPoints le contenu de nextPoints
		prevPoints = nextPoints;
		// - > si prevPoints ne contient pas assez de points,
		// appeler la fonction de detection de points sur prevInput
		if (prevPoints.size() <= 10) {
			detectPoints(prevInput);
			// - > si prevPoints ne contient toujours pas assez de points,
			// quitter la fonction
			if (prevPoints.size() <= 10) return;
		}
		// - > calculer a l’aide de calcOpticalFlowPyrLK les nouvelles positions
		// (nextPoints) dans la nouvelle image (nextImage)
		std::vector<uchar> status;
		std::vector<float> err;
		cv::calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, status, err);
		// -> supprimer les points non suivis des deux listes de points
		prevPoints = purgePoints(prevPoints, status);
		nextPoints = purgePoints(nextPoints, status);
		updateROI();
		//cloner nextInput dans prevInput
		prevInput = nextInput.clone();
	}
	else {
		std::cout << "rien a track" << std::endl;
	}
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		start = cv::Point(x, y);
		roi = cv::Rect();
		prevPoints.clear();
		nextPoints.clear();
	}
	else if (event == cv::EVENT_MOUSEMOVE)
	{
		if (start.x >= 0) {
			cv::Point end(x, y);
			roi = cv::Rect(start, end);
		}
	}
	else if (event == cv::EVENT_LBUTTONUP) {
		cv::Point end(x, y);
		roi = cv::Rect(start, end);
		start = cv::Point(-1, -1);
	}
}

void video(char* videoname) {
	cv::VideoCapture cap;
	//si videoname n’est pas null, ouvrir la video dans cap, sinon ouvrir la camera 0
	videoname != nullptr ? cap.open(videoname) : cap.open(0);
	//si cap n’est pas ouvert, quitter la fonction
	if (!cap.isOpened()) return;
	//recuperer une image depuis cap et la stocker dans nextInput
	cap >> nextInput;
	prevInput = nextInput;
	//tant que nextinput n’est pas vide
	while (!nextInput.empty()) {
		// - > faire les traitements sur l’image (prochaines étapes)
		trackPoints();
		//detectPoints(nextInput);
		// - > appeler la fonction de dessin
		draw();
		// - > recuperer une nouvelle image et la stocker dans nextInput
		cap >> nextInput;
		// - > attendre 10ms que l’utilisateur tape une touche, et quitter si il le fait
		if (cv::waitKey(10) >= 0) break;
	}
}

int main()
{

	std::cout << "Lancement de la video ou camera" << std::endl;

	video(nullptr);

	//system("PAUSE");
}