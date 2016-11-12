#include "Kalman.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


KalmanFilterTracker::KalmanFilterTracker() //Tracker attributes
{
	//ctor
	KF = KalmanFilter(4, 2, 0);
	state = Mat(4, 1,CV_32F);
	precessNoise = Mat(4, 1, CV_32F);	//precession noise - variance
	measurement = Mat(2, 1,CV_32F);
	//measurement.setTo(Scalar(0));
	init = true;
	tracked = true;
	matched = false;	//intiialisation, active status, matching flags
	unmatchedFrames = 0;

}

////////////Initialise Kalman Filter/////////////////
//using matched blob attributes as measurements
void KalmanFilterTracker::initilizeKF(Point center, double dT) 
{

	KF.statePre.at<float>(0) = center.x;
	KF.statePre.at<float>(1) = center.y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-2));//initially 1e-2
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
	setIdentity(KF.errorCovPost, Scalar::all(.2));//initially .1

	points_vector.clear();
	kalman_vector.clear();
	init = false;	//drops initialisation flag
	//tracked = true;
}

//////////////////////////////Tracking - state estimation and measruement correction//////////////////////////////////////
void KalmanFilterTracker::track(Point center, double dT) {

	if (init)
		initilizeKF(center,dT);

	state = KF.predict();	//performs state estimation
	
	
	measurement.at<float>(0) = center.x;
	measurement.at<float>(1) = center.y;	//updates measurement matrix with blob centerpoint

	Point measPt(measurement.at<float>(0), measurement.at<float>(1));
	points_vector.push_back(measPt);

	Mat estimated = KF.correct(measurement);	//perform measurement update
	Point statePt(estimated.at<float>(0), estimated.at<float>(1));
	kalman_vector.push_back(statePt);

}
