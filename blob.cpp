// Blob.cpp

#include "blob.h"
using namespace cv;
using namespace std;
///////////////////////////////////////////////////////////////////////////////////////////////////
Blob::Blob(vector<Point> _contour) //Blob object attributes
{

	currentContour = _contour;

	boundingRect = cv::boundingRect(currentContour);	//finds minimum bounding rectangle around blob contour


	centerPosition.x = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;	
	centerPosition.y = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;	//centerpoint of blob

	
	diagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));	

	aspectRatio = (float)boundingRect.width / (float)boundingRect.height;

	matched = false;	//matching flag
	}

///////////////////////////////////////////////////////////////////////////////////////////////////