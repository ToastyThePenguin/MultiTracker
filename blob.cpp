// Blob.cpp

#include "blob.h"
using namespace cv;
using namespace std;
///////////////////////////////////////////////////////////////////////////////////////////////////
Blob::Blob(vector<Point> _contour) //Blob object attributes
{

	currentContour = _contour;

	currentBoundingRect = cv::boundingRect(currentContour);	//finds minimum bounding rectangle around blob contour

	cv::Point currentCenter;

	centerPosition.x = (currentBoundingRect.x + currentBoundingRect.x + currentBoundingRect.width) / 2;	
	centerPosition.y = (currentBoundingRect.y + currentBoundingRect.y + currentBoundingRect.height) / 2;	//centerpoint of blob

	
	dblCurrentDiagonalSize = sqrt(pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2));	

	dblCurrentAspectRatio = (float)currentBoundingRect.width / (float)currentBoundingRect.height;

	matched = false;	//matching flag
	
	intNumOfConsecutiveFramesWithoutAMatch = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////