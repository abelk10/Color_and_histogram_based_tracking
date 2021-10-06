#include "tracker.hpp"
using namespace tracker;
using namespace std;
using namespace cv;

Track::Track(){
	_model_init = false;
	_delta = 10;
	_bins = 16;
	_candid = 9;
	_hist_w = 512, _hist_h = 400;
	_hist_plot = Mat(_hist_h, _hist_w, CV_8UC3, Scalar( 0,0,0));
}
Track::~Track(){};

Mat Track::getModelROI(void){
	return extractRegion(_model_frame,_model_bbox);
}

void Track::setModel(Mat& frame, Rect& bbox, int numCandidates, int delta, int bins ){
	frame.copyTo(_model_frame);
	_model_bbox = bbox;
	_pred_center.push_back(Point(_model_bbox.x+_model_bbox.width/2,_model_bbox.y+_model_bbox.height/2));
	_delta = delta;
	_bins = bins;
	_candid = numCandidates;
	_model_init = true;
}

bool Track::modelInitialized(void){
	return _model_init;
}

void Track::resetTracker(void){
	_model_init = false;
}

Point Track::rectToCenter(Rect rect){
	return Point(rect.x+rect.width/2,rect.y+rect.height/2);
}

Rect Track::centerToRect(Point center){
	// Set x coordinate of upper left corner of rectangle to zero if it's out of the frame(negative)
	int x0 = max(center.x-_model_bbox.width/2, 0);
	// Adjust x coordinate of upper left corner of rectangle to prevent bounding box being out of frame
	// in the right side
	if(x0+_model_bbox.width >=  _model_frame.cols){
		x0 = _model_frame.cols - _model_bbox.width;
	}
	// Set y coordinate of upper left corner of rectangle to zero if it's out of the frame(negative)
	int y0 = max(center.y-_model_bbox.height/2, 0);
	// Adjust y coordinate of upper left corner of rectangle to prevent bounding box being out of frame
	// in the bottom side
	if(y0+_model_bbox.height >=  _model_frame.rows){
		y0 = _model_frame.rows - _model_bbox.height;
	}
	return Rect(x0, y0,_model_bbox.width,_model_bbox.height);
}
double Track::distance(Mat& Qu, Mat& Pu){return 0;}
Mat Track::features(Mat& frame, HIST_TYPE type){
	Mat temp(1, 1, CV_32FC1, Scalar(0));
	return temp;
}


Mat Track::extractRegion(Mat& frame, Rect& bbox){
	Mat roi = _model_frame(_model_bbox);
	return roi;
}

vector<Rect> Track::candidateRegions(Mat& frame, Point center){
	int n = (int)sqrt(_candid);
	_candidates.clear();	// Clear candidates from previous frame
	// if n is odd then we have equal number candidate pixels in all directions
	// but if n is even the we will have one more candidate in either side
	// hence we add (1-n%2) on the right and bottom side
	for(int i=-(n-1)/2;i<=(n-1)/2+(1-n%2);i++){
		for(int j=-(n-1)/2;j<=(n-1)/2+(1-n%2);j++){
			_candidates.push_back(centerToRect(Point(center.x+(_delta*i), center.y+(_delta*j))));
		}
	}
	return _candidates;
}

Rect Track::track(Mat& frame, HIST_TYPE type){
	candidateRegions(frame, _pred_center[_pred_center.size()-1]);	// Generate candidates
	Mat candid_features, model_features, roi;
	Mat model_region = getModelROI();	// Get model ROI
	model_features = features(model_region, type);	// Extract ROI color feature
	Mat distances(1, _candidates.size(), CV_32FC1, Scalar( 0,0,0));
	for(int i=0; i<int(_candidates.size()); i++){
		roi = frame(_candidates[i]);	// Extract candidate ROI from frame
		candid_features = features(roi, type);	// Extract features for candidate
		distances.at<float>(i) = distance(model_features, candid_features);	// Compute distance
	}
	double min, max;
	Point min_loc, max_loc;
	minMaxLoc(distances, &min, &max, &min_loc, &max_loc);	// Find minimum distance
	_pred_center.push_back(rectToCenter(_candidates[min_loc.x]));	// Add new prediction center that has minimum distance
	return centerToRect(_pred_center[_pred_center.size()-1]);	// Return tracking prediction bounding box
}

//	ColorTrack methods
ColorTrack::ColorTrack(){}
ColorTrack::~ColorTrack(){};
Mat ColorTrack::getHistPlot(void){
	return _hist_plot;
}

Mat ColorTrack::getHeatMap(void){
	return _heat_map;
}

void ColorTrack::histPlot(Mat histograms, int size){
	_hist_plot.setTo(Scalar( 0,0,0));	// Clear histogram plot
	int bin_w = cvRound( (double) _hist_w/size);
	for( int i = 1; i < size; i++ )
	{
		line( _hist_plot, Point(bin_w*(i-1), _hist_h - cvRound(histograms.at<float>(i-1)*_hist_h)),
			  Point(bin_w*(i), _hist_h - cvRound(histograms.at<float>(i)*_hist_h)),
			  Scalar(0, 255, 0), 2, 8, 0);
	}
}

void ColorTrack::heatMap(Mat& frame, HIST_TYPE type){
	Mat frame_hsv, frame_gray;
	vector<Mat> bgr_planes, hsv_planes;

	//Convert frame to grayscale and HSV
	cvtColor(frame, frame_hsv, CV_BGR2HSV);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	split(frame, bgr_planes);
	split(frame_hsv, hsv_planes);
	int t = 2;
	switch(type){
	case HUE:
		applyColorMap(hsv_planes[0], _heat_map, t);
		break;
	case SATURATION:
		applyColorMap(hsv_planes[1], _heat_map, t);
		break;
	case RED:
		applyColorMap(bgr_planes[2], _heat_map, t);
		break;
	case GREEN:
		applyColorMap(bgr_planes[1], _heat_map, t);
		break;
	case BLUE:
		applyColorMap(bgr_planes[0], _heat_map, t);
		break;
	case GRAY:
		applyColorMap(frame_gray, _heat_map, t);
		break;
	case ALL:
		break;
	}
}

double ColorTrack::distance(Mat& Qu, Mat& Pu){
	double distance = compareHist(Qu, Pu, CV_COMP_BHATTACHARYYA);
	return distance;
}

Mat ColorTrack::features(Mat& frame, HIST_TYPE type){

	Mat frame_hsv, frame_gray;
	vector<Mat> bgr_planes, hsv_planes;
	Mat histograms;

	//Convert frame to grayscale and HSV
	cvtColor(frame, frame_hsv, CV_BGR2HSV);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);

	//Split BGR and HSV channels
	split(frame, bgr_planes);
	split(frame_hsv, hsv_planes);

    Mat b_hist, g_hist, r_hist, h_hist, s_hist, gray_hist;
    float range[] = {0, 256};		//Range of values for all channels except for Saturation
    const float* Range = { range };
    float s_range[] = { 0, 180};	//range of values for Saturation
    const float* sRange = { s_range };

    if(type==RED || type==ALL){
    	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &_bins, &Range);
		normalize(r_hist, r_hist, 1, 0, NORM_L2);
		histograms.push_back(r_hist);
    }
    if(type==GREEN || type==ALL){
    	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &_bins, &Range);
		normalize(g_hist, g_hist, 1, 0, NORM_L2);
		histograms.push_back(g_hist);
	}
    if(type==BLUE || type==ALL){
    	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &_bins, &Range);
		normalize(b_hist, b_hist, 1, 0, NORM_L2);
		histograms.push_back(b_hist);
	}
    if(type==HUE || type==ALL){
    	calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &_bins, &Range);
		normalize(h_hist, h_hist, 1, 0, NORM_L2);
		histograms.push_back(h_hist);
	}
    if(type==SATURATION || type==ALL){
    	calcHist( &hsv_planes[1], 1, 0, Mat(), s_hist, 1, &_bins, &sRange);
		normalize(s_hist, s_hist, 1, 0, NORM_L2);
		histograms.push_back(s_hist);
	}
    if(type==GRAY || type==ALL){
    	calcHist( &frame_gray, 1, 0, Mat(), gray_hist, 1, &_bins, &Range);	//Grayscale histogram and normalization
		normalize(gray_hist, gray_hist, 1, 0, NORM_L2);
		histograms.push_back(gray_hist);
	}
    histPlot(histograms, histograms.rows);		// Generate histogram plot

	return histograms;

}
Rect ColorTrack::track(Mat& frame, HIST_TYPE type){
	heatMap(frame, type);
	return Track::track(frame, type);
}

//	GradientTrack methods

GradientTrack::GradientTrack(){}
GradientTrack::~GradientTrack(){};
double GradientTrack::distance(Mat& Qu, Mat& Pu){
	double distance = compareHist(Qu, Pu, CV_COMP_CHISQR);
	return distance;
}

Mat GradientTrack::features(Mat& frame, HIST_TYPE type){

	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);

	vector<float> features;
	vector<Point> locations;
	HOGDescriptor hog(Size(64,128), Size(16,16), Size(8,8), Size(8,8), _bins, 1);
	resize(frame_gray, frame_gray, Size(64, 128));
	hog.compute(frame_gray, features);

	Mat features_mat(features.size(), 1, CV_32FC1, Scalar(0));

	for(int i=0; i<int(features.size());i++){
		features_mat.at<float>(i) = features[i];
	}
    return features_mat;

}

// FusionTrack methods
FusionTrack::FusionTrack(){}
FusionTrack::~FusionTrack(){};

void FusionTrack::setModel(Mat& frame, Rect& bbox, int numCandidates1, int delta1, int bins1, int numCandidates2, int delta2, int bins2, bool color, bool gradient){
	Track::setModel(frame, bbox, numCandidates1, delta1, bins1);
	_color = color;
	_gradient = gradient;
	_ct.setModel(frame, bbox, numCandidates1, delta1, bins1);
	_gt.setModel(frame, bbox, numCandidates2, delta2, bins2);
}

Mat FusionTrack::features(Mat& frame, HIST_TYPE type){
	Mat temp(1, 1, CV_32FC1, Scalar(0));
	return temp;
}

double FusionTrack::distance(Mat& Qu, Mat& Pu){return 0;}

Rect FusionTrack::track(Mat& frame, HIST_TYPE type){
	candidateRegions(frame, _pred_center[_pred_center.size()-1]);	// Generate candidates
	Mat candid_features, model_features_ct, model_features_gt, roi;
	Mat model_region = getModelROI();	// Get model ROI
	if(_color){
		model_features_ct = _ct.features(model_region, type);	// Extract ROI color feature
	}
	if(_gradient){
		model_features_gt = _gt.features(model_region, type);	// Extract ROI gradient feature
	}
	Mat allDistances(2, _candidates.size(), CV_32FC1, Scalar( 0,0,0));
	for(int i=0; i<int(_candidates.size()); i++){
		roi = frame(_candidates[i]);	// Extract candidate ROI from frame
		if(_color){
			candid_features = _ct.features(roi, type);	// Extract features for candidate
			allDistances.at<float>(0,i) = _ct.distance(model_features_ct, candid_features);	// Compute color feature distance
		}
		else{
			allDistances.at<float>(0,i) = 0;
		}
		if(_gradient){
			candid_features = _gt.features(roi, type);	// Extract features for candidate
			allDistances.at<float>(1,i) = _gt.distance(model_features_gt, candid_features);	// Compute gradient feature distance
		}
		else{
			allDistances.at<float>(1,i) = 0;
		}
	}
	Mat distances;
	normalize(allDistances.row(0), allDistances.row(0), 1, 0, NORM_L2);	//	Normalize color feature distances
	normalize(allDistances.row(1), allDistances.row(1), 1, 0, NORM_L2); //	Normalize gradient feature distances
	reduce(allDistances, distances, 0, CV_REDUCE_SUM, CV_32FC1);	// Fuse color and gradient feature distances for canidates
	double min, max;
	Point min_loc, max_loc;
	minMaxLoc(distances, &min, &max, &min_loc, &max_loc);	// Find minimum distance
	_pred_center.push_back(rectToCenter(_candidates[min_loc.x]));	// Add new prediction center that has minimum distance
	return centerToRect(_pred_center[_pred_center.size()-1]);	// Return tracking prediction bounding box
}
