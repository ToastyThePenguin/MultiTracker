////////////////////////MultiTracker////////////////////////////
//////////Designed and optimised for bat tracking//////////////
//
//#
// ##
// ###
//  ####
//   #####
//   #######
//    #######
//    ########
//    #########
//   ############
//################
// ################												   ##
//  ##############												 ###
//   ##############                                            ####
//    ##############                                         #####
//   ##############                                      #######
//   ##############                                 ###########
//  ################                            ##############
//  #################       #                  ################
//  ##################     ##    #           #################
// ####################   ###   ##          #################
//      ################  ########          #################
//       ################  #######         ###################
//         #######################       #####################
//          #####################       ###################
//            ############################################
//            ###########################################
//             ##########################################
//              ########################################
//               ######################################
//               ######################################
//                ##########################      #####
//                ###  ###################           ##
//                ##    ###############
//                #     ##  ##########   
//                      ##    ###
//                            ###
//                            ##
//                            #
//
/////////////DEPENDENCY INCLUDES///////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include "Kalman.h"	//Kalman filter class header file
#include "blob.h"	//Blob class header file
#include <vector>
#include <conio.h>
#include <sstream>

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


//////////////////////////Colour Constants///////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_CYAN = cv::Scalar(255.0, 255.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_PURPLE = cv::Scalar(204.0, 0.0, 102.0);

////////////////////Global Variables/////////////////////////////

Ptr <BackgroundSubtractorMOG2> PMOG2;//background subtractor object
int frameCount = 0; //stores the total number of frames 
int objectsCount = 0;	//stores the number of objects identified as blobs
double dT = 0;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

using namespace cv;
using namespace std;

/////////////////////////////////////////Function Prototypes//////////////////////////////////////////////////////////////////////
bool lineCrossed(std::vector<KalmanFilterTracker> &trackers, int &intLeftLinePosition, int &intRightLinePosition, int &carCount);
void drawTrackers(vector<KalmanFilterTracker>&trackers, Mat img, vector<Blob> &blobs);
void drawBatCount(int &batCount, Mat img);
double distanceBetweenPoints(Point point1, Point point2);
double distY(Point point1, Point point2);
void matchBlobstoTrackers(vector<Blob> &currentFrameBlobs,vector<KalmanFilterTracker> &trackers);
void drawContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

///////////////////////////////////Main Function//////////////////////////////////////////////////////////////////////////////

int main(void)
{

	//////////////////////////////create variables//////////////////////////////////

	cv::VideoCapture capVideo;	//object to take in dataset video. Not necessary if RealSense is being used with live feed
	
	////////////////////////////Uncomment if results capture is desired///////////////////////////////////
	//cv::VideoWriter vidResult("result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 60, Size(640, 480), true);

	PMOG2 = createBackgroundSubtractorMOG2();
	cv::Mat img;
	Point middle = (320, 240);
	vector<KalmanFilterTracker> trackers;	//vector array of Kalman filter objects for tracking
	
	cv::Point vertLine[2];	//stores start and end point for crossing lines used in counting
	cv::Point horizLine[2];

	int batCount = 0;
	capVideo.open("colour.avi");	//opens dataset video, set to 0 if stream from webcam is being used

	////////////////////////Safety precaution in case of capture failure/////////////////////////////
	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "error reading video file" << std::endl << std::endl;      // show error message
		_getch();                   
		return(0);                                                              // and exit program
	}

	/////////////////////////Validity check to ensure program can at least run once//////////////////
	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		std::cout << "error: video file must have at least two frames";
		_getch();       
		return(0);
	}


	capVideo >> img;	//obtain frame from capture and store in matrix
	

	///////////////Define parameters of crossing lines for counting////////////////
	int intHorizLinePosition = (int)std::round((double)img.rows * 0.6);	//positions horizontal line at a position 3/5 down the total length of the y-axis
	int intVertLinePosition = (int)std::round((double)img.cols * 0.7);	//positions horizontal line at a position 7/10 down the total length of the x-axis

	horizLine[0].y = intHorizLinePosition;
	horizLine[0].x = 0;
	horizLine[1].x = img.cols - 1;
	horizLine[1].y = intHorizLinePosition;


	vertLine[0].x = intVertLinePosition;
	vertLine[0].y = 0;
	vertLine[1].y = img.rows - 1;
	vertLine[1].x = intVertLinePosition;

	/////////////////////////////////////////////////////////////////////////////
	
	char esc = 0;	//stores latest key press to be used for checking if the escape key has been pressed

	bool blnFirstFrame = true;

	for (int i = 0; i < 10; i++) capVideo >> img;	//cycles through first 10 frames to filter out initialisation noise
	
	frameCount = 11;

	double ticks = 0;
	clock_t start = clock();
	
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////


	/////////////////////////////////MAIN LOOP////////////////////////////////////
	while (capVideo.isOpened() && esc != 27)
	{
		
		//////////Structuring elements of various sizes for use in motphological transforms///////////////
		cv::Mat structuringElement2x2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
		cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::Mat structuringElement10x10 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));

		std::vector<Blob> currentFrameBlobs;	//stores the contours inthe current frame identified as targets

		cv::Mat result = img.clone();	//stores a copy of the input image for later use

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		cv::cvtColor(img, img, CV_BGR2GRAY);	//converts input frame to grayscale, disable this line when using infrared datasets
		
		PMOG2->apply(img, imgDifference, -1);	//apply background subtraction and obtain foreground mask
		//imshow("Difference1", imgDifference);	//display difference mask (used during development and debugging)
		
		erode(imgDifference, imgDifference, structuringElement5x5);
		erode(imgDifference, imgDifference, structuringElement5x5);	//double erosion to remove noise
		//erode(imgDifference, imgDifference, structuringElement7x7);


		cv::threshold(imgDifference, imgThresh, 20, 255.0, CV_THRESH_BINARY);	//performs thresholdingand obtains threshold mask

		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		dilate(imgThresh, imgThresh, structuringElement7x7);		
		dilate(imgThresh, imgThresh, structuringElement10x10);	//3-stage dilation to fill out contours and facilitate easier detection
		//cv::imshow("Threshold2", imgThresh);	//display threshold mask (used during development and debugging)
		std::vector<std::vector<cv::Point> > contours;	//vector array for storing current frame contours

		cv::findContours(imgThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);	//finds external contours i.e. outermost boundaries

		
		std::vector<std::vector<cv::Point> > convexHulls(contours.size());	//vector array for storing convex hull contours

		for (unsigned int i = 0; i < contours.size(); i++) 
		{
			cv::convexHull(contours[i], convexHulls[i]);
		}	//converts all contours into convex hulls which fills in holes

		//drawContours(imgThresh.size(), convexHulls, "imgConvexHulls");	//displays contours (used in development and debugging)

		//////////////////Stage 1:	Find targets in current frame///////////////////
		for (auto &convexHull : convexHulls) //iterates through all current frame contours
		{
			Blob possibleBlob(convexHull);	//creates a blob object from the current contour

			if (possibleBlob.currentBoundingRect.area() > 400 &&
				possibleBlob.dblCurrentAspectRatio > 0.5 &&
				possibleBlob.dblCurrentAspectRatio < 4.0 &&
				possibleBlob.currentBoundingRect.width > 20 &&
				possibleBlob.currentBoundingRect.height > 20 &&
				possibleBlob.dblCurrentDiagonalSize > 30.0 &&
				(cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.5) //minimum qualifying crriteria for targets, this helps eliminate false positives
			{
				currentFrameBlobs.push_back(possibleBlob);	//adds contour to array of current frame blobs if it meets criteria

			}
		}

		//drawContours(imgThresh.size(), currentFrameBlobs, "Current Frame Blobs");	//displays current frame blobs (used for development and debuging)

		////////////////Stage 2:	perform matching, counting and display results/////////////////
		
		matchBlobstoTrackers(currentFrameBlobs,trackers);	//calls mathcing function, takes arrays of current frame blobs and kalman filter objects as arguments
		
		drawTrackers(trackers, result, currentFrameBlobs);	//once matching has been performed display the trackers (Kalman filters) still active

		bool blnAtLeastOneBlobCrossedTheLine = lineCrossed(trackers, intHorizLinePosition, intVertLinePosition, batCount);	//calls line crossing function, returns true or false

		if (blnAtLeastOneBlobCrossedTheLine == true) //line goes green if it has been crossed
		{
			//cv::line(result, horizLine[0], horizLine[1], SCALAR_GREEN, 2);
			cv::line(result, vertLine[0], vertLine[1], SCALAR_GREEN, 2);

		}
		else //otherwise line remains red
		{
			//cv::line(result, horizLine[0], horizLine[1], SCALAR_RED, 2);
			cv::line(result, vertLine[0], vertLine[1], SCALAR_RED, 2);

		}

		drawBatCount(batCount, result);	//displays the object count on the result frame

		cv::imshow("Result", result);	//display results with line, active trackers, marked targets, frame count and bat count
		std::cout << "Trackers: " << trackers.size() << endl;


	/////////enable this to capture results in video/////////
/*
		for (int i = 0; i < 6; i++)	//frames duplicated to slow down footage and make results easierto observe 
		{
			vidResult << result;
		}
*/
		//cv::waitKey(0);      // enable to step through frame by frame (used for debugging and evaluation)
										
		//////////////////////Housekeeping////////////////
		int trackersCount = trackers.size();	//get number of trackers
		objectsCount += currentFrameBlobs.size();	//update number of detected objects
		currentFrameBlobs.clear();	//clear vector of current blobs

		/////////////////////////////////grab next frame in sequence provided end of video has not been reached///////////////////////
		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
			capVideo.read(img);
			//img.convertTo(img, CV_8UC1);
		}
		////////////////////////If end of video has been reached display number of trackers, frames, total objects detected//////////////////////////
		else    
		{
			std::cout << "end of video\n";
			cout << "Frames: "<<frameCount << endl;
			cout << "Trackers: "<<trackersCount << endl;
			cout << "Objects: "<<objectsCount << endl;
			cout << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
			break;
		}

		blnFirstFrame = false;
		frameCount++;	//increment frame count
		esc = cv::waitKey(1);	//accept input from keyboard
	}

	if (esc != 27) //checks if escape key has been pressed
	{               
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
			}


	
	capVideo.release();
	//vidResult.release();
	batCount = 0;
	trackers.clear();	 //clear vector of trackers

	return 0;
}

/////////////////////////////////////////////////////////////////////FIN////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// <<<<<<<<<<<<<<<<<<<Check if line has been crossed>>>>>>>>>>>>>>>>>>>>>>>>
bool lineCrossed(std::vector<KalmanFilterTracker> &trackers, int &intLeftLinePosition, int &intRightLinePosition, int &batCount)
{
	bool CrossedTheLine = false;	//creates line crossing flag initialised to 0

	for (auto tracker : trackers) //iterates through all trackers
	{

		if (tracker.matched == true && tracker.kalman_vector.size() >= 2) //only considers currently matched trackers to ensure strongr correlation with targets and ground truth
		{
			int prevFrameIndex = (int)tracker.kalman_vector.size() - 2;
			int currFrameIndex = (int)tracker.kalman_vector.size() - 1;

			//increments count and raises line crossing flag if previous and current tracker positions lie on opposite side of the crossing line 
			if (tracker.kalman_vector[prevFrameIndex].x > intRightLinePosition && tracker.kalman_vector[currFrameIndex].x <= intRightLinePosition) 
			{
				batCount++;
				CrossedTheLine = true;
			}
			//ensures crossing works both ways
			else if (tracker.kalman_vector[prevFrameIndex].x < intRightLinePosition && tracker.kalman_vector[currFrameIndex].x >= intRightLinePosition)
			{
				batCount++;
				CrossedTheLine = true;

			}
			else {}
		}

	}

	return CrossedTheLine;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// <<<<<<<<<<<<<<<<<<<Draw trackers on image>>>>>>>>>>>>>>>>>>>>>>>>
void drawTrackers(vector<KalmanFilterTracker> &trackers, Mat img, vector<Blob> &blobs)
{
	for (unsigned int i = 0; i < trackers.size(); i++) {
		if (trackers[i].tracked == true && trackers[i].points_vector.size() >= 4) {
			for (unsigned int j = 0; j < trackers[i].kalman_vector.size() - 1; j++)
				line(img, trackers[i].kalman_vector[j], trackers[i].kalman_vector[j + 1], SCALAR_CYAN, 2);

			int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			double dblFontScale = 1.0;
			int intFontThickness = (int)std::round(dblFontScale * 1.0);

			cv::putText(img, std::to_string(i), trackers[i].kalman_vector.back(), intFontFace, dblFontScale, SCALAR_RED, intFontThickness);
		}
	}

	for (unsigned int i = 0; i < blobs.size(); i++) {

		cv::rectangle(img, blobs[i].currentBoundingRect, SCALAR_PURPLE, 2);
	}


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// <<<<<<<<<<<<<<<<<<<Draw car count on image>>>>>>>>>>>>>>>>>>>>>>>>

void drawBatCount(int &batCount, Mat img)
{

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (img.rows * img.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	cv::Size textSize = cv::getTextSize(std::to_string(batCount), intFontFace, dblFontScale, intFontThickness, 0);

	cv::Point ptTextBottomLeftPosition;
	cv::Point ptTextTopLeftPosition;

	ptTextBottomLeftPosition.x = img.cols - 1 - (int)((double)textSize.width * 1.25);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);
	ptTextTopLeftPosition.x = (int)((double)textSize.width );
	ptTextTopLeftPosition.y = (int)((double)textSize.height);

	cv::putText(img, std::to_string(batCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
	cv::putText(img, std::to_string(frameCount), ptTextTopLeftPosition, intFontFace, dblFontScale/2.0, SCALAR_RED, intFontThickness/1.5);
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// <<<<<<<<<<<<<<<<<<<Euclidean Distance>>>>>>>>>>>>>>>>>>>>>>>>

double distanceBetweenPoints(Point point1, Point point2)
{

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);
	return sqrt(pow(intX,2)+ pow(intY, 2));
	//return intY;	//only difference in y positions can be returned if desired
}
///////////////////////////////////////////////////////////////

double distY(Point point1, Point point2)
{

	double X = abs(point1.x - point2.x);
	double Y = abs(point1.y - point2.y);
	return Y;
}
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////



// <<<<<<<<<<<<<<<<<<<It's all in the name>>>>>>>>>>>>>>>>>>>>>>>>
void matchBlobstoTrackers(vector<Blob> &currentFrameBlobs, vector<KalmanFilterTracker> &trackers)
{
	for (auto &tracker : trackers)
	{
		tracker.matched = false;	//sets all trackers to unmatched
	}
	for (auto &blob : currentFrameBlobs)	//iterates through all current frame blobs to find matches
	{
		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;
		double Y = 10000.0;
		for (int  i =0; i < trackers.size(); i++)	//iterates through all trackers to compare them with the current blob
		{

			int currIndex = trackers[i].kalman_vector.size() - 1;
			int prevIndex = trackers[i].kalman_vector.size() - 2;
			if (trackers[i].tracked == true && trackers[i].matched ==false)	//trackers areonly compared to blobs if they are unmatched and active
			{
				double distance = distanceBetweenPoints(trackers[i].kalman_vector.back(), blob.centerPosition);	//determines Euclidean distance between blob centroid and latest predicted position of tracker
				int diff1 = (blob.centerPosition.x - trackers[i].kalman_vector[currIndex].x);//determines vector direction of current blob and latest predicted position for tracker
				int diff2 = (trackers[i].kalman_vector[currIndex].x - trackers[i].kalman_vector[prevIndex].x);	//determines vectore direction of last two predicted positions of tracker
				
				if (diff1 >=0 && diff2>=0)	//uses vector directions to ensure that blob lies on tracker's trajectory
				{
					//if Euclidean distance is less than the currently stored shortest distance to the blob's centroid the Euclidean distance replaces it
					//the index of the tracker with the new shortest distance is stored
					if (distance < dblLeastDistance) 
					{
						dblLeastDistance = distance;
						intIndexOfLeastDistance = i;
						Y = distY(trackers[i].kalman_vector.back(), blob.centerPosition);
					}
				}

				//repeat of the above but for movement in the opposite direction
				if (diff1<=0 && diff2<=0)
				{
					if (distance < dblLeastDistance) {
						dblLeastDistance = distance;
						intIndexOfLeastDistance = i;
						Y = distY(trackers[i].kalman_vector.back(), blob.centerPosition);
					}
				}

			}

			
		}
		//If tracker is close enoughto current blob the two are matched
		//Tracker measurements are updated with blob attributes
		//Next posiiton is then predicted
		if (dblLeastDistance<blob.dblCurrentDiagonalSize*2.0) 
		{
			trackers[intIndexOfLeastDistance].matched = true;
			trackers[intIndexOfLeastDistance].unmatchedFrames = 0;
			
			trackers[intIndexOfLeastDistance].track(blob.centerPosition, dT);
			blob.matched = true;
		}

		//if no match could be found for the current blob a new tracker is created and initialised
		//Then added to array of trackers
		if (blob.matched == false)	 
		{
			KalmanFilterTracker newTracker;
			newTracker.initilizeKF(blob.centerPosition,dT);
			newTracker.track(blob.centerPosition,dT);
			trackers.push_back(newTracker);
		}

	}

	for (auto &tracker : trackers)	//iterates through all existing Kalman filters 
	{
		//predicts next position for unmatched trackers using previous prediction
		//increments number of consecutive frames without a match
		if (tracker.matched == false) 
		{
			tracker.track(tracker.kalman_vector.back(),dT);
			tracker.unmatchedFrames++;
		}

		//when number of consecutive frames without a match exceeds defined thresholdtracker is deactivated 
		if (tracker.unmatchedFrames >= 2) {
			tracker.tracked = false;

			
		}

	}

}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<Draw Contours>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void drawContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) //draws current frame contours or convex hulls
{
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//void drawContours(cv::Size imageSize, std::vector<KalmanFilterTracker> trackers, std::string strImageName) {
//
//	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
//
//	std::vector<std::vector<cv::Point> > contours;
//
//	for (auto &tracker : trackers) {
//		if (tracker.tracked == true) {
//			contours.push_back(blob.currentContour);
//		}
//	}
//
//	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
//
//	cv::imshow(strImageName, image);
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) //draws current frame blobs
{

	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	std::vector<std::vector<cv::Point> > contours;

	for (auto &blob : blobs) {
			contours.push_back(blob.currentContour);
		
	}

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
