#ifndef KALMANFILTERTRACKER_H
#define KALMANFILTERTRACKER_H

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/video/tracking.hpp>
#include <vector>

using namespace cv;
using namespace std;

class KalmanFilterTracker
{
public:
	KalmanFilter KF;
	Mat state;
	Mat precessNoise;
	Mat measurement;
	vector<Point>points_vector; 
	vector <Point> kalman_vector;
	bool init;
	
	bool tracked;
	bool matched;
	int matchedFrames;
	int unmatchedFrames;
	void track(Point center,double dT);
	void initilizeKF(Point center,double dT);
	Mat draw(Mat img);
	KalmanFilterTracker(void);
};

#endif // KALMANFILTERTRACKER_H
