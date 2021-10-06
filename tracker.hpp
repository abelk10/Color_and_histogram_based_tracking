
#ifndef TRACKER_H_INCLUDE
#define TRACKER_H_INCLUDE
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
typedef enum {
	RED=0,
	GREEN=1,
	BLUE=2,
	HUE=3,
	SATURATION=4,
	GRAY=5,
	ALL=6
} HIST_TYPE;


namespace tracker{
	class Track{
	protected:
		// Initial model frame and bounding box
		Mat _model_frame;
		Rect _model_bbox;

		bool _model_init; // boolean to check if model has been initialized with frame
		int _delta; // delta value for candidate generation
		int _candid; // number of candidates to generate
		int _hist_w, _hist_h; // histogram plot width and height
		vector<Point> _pred_center;	// A list(vector) of prediction centers
		Mat _hist_plot;	//	Plot of histogram to display
		vector<Rect> _candidates;	//	A list(vector) for generated candidate bounding boxes
		int _bins;	// Number of histogram bins

	public:
		Track();
		virtual ~Track();
		Mat getModelROI(void);	// Extract region of interest from model frame
		void setModel(Mat& frame, Rect& bbox, int numCandidates=9, int delta=3, int bins=16);	// Set model frame and bounding box for tracking
		bool modelInitialized(void);	// Check if model is initialized with model frame
		void resetTracker(void);	// reset tracker for new object

		Rect centerToRect(Point center);	// Generate bounding box from center position
		Point rectToCenter(Rect rect);	// Get center of bounding box


		virtual double distance(Mat& Qu, Mat& Pu);
		virtual Mat features(Mat& frame, HIST_TYPE type);
		Mat extractRegion(Mat& frame, Rect& bbox);	//Extract bouding box region from frame
		vector<Rect> candidateRegions(Mat& frame, Point center);

		Rect track(Mat& frame, HIST_TYPE type);
	};

	class ColorTrack: public Track{
		Mat _heat_map; //	Heat map of frame

	public:

		ColorTrack();
		virtual ~ColorTrack();
		double distance(Mat& Qu, Mat& Pu);
		Mat features(Mat& frame, HIST_TYPE type);
		Rect track(Mat& frame, HIST_TYPE type);

		void heatMap(Mat& frame, HIST_TYPE type);
		Mat getHistPlot(void);	// Get histogram image
		Mat getHeatMap(void);	// Get frame heatmap
		void histPlot(Mat histograms, int size);	// Generate histogram plot
	};

	class GradientTrack: public Track{

	public:

		GradientTrack();
		virtual ~GradientTrack();
		double distance(Mat& Qu, Mat& Pu);
		Mat features(Mat& frame, HIST_TYPE type=ALL);
	};

	// Fusion of color and Gradient trackers
	class FusionTrack: public Track{
		ColorTrack _ct;
		GradientTrack _gt;
		bool _color, _gradient;

	public:

		FusionTrack();
		virtual ~FusionTrack();
		void setModel(Mat& frame, Rect& bbox, int numCandidates1=9, int delta1=3, int bins1=16, int numCandidates2=9, int delta2=3, int bins2=16, bool color=true, bool gradient=true);	// Set model frame and bounding box for tracking
		double distance(Mat& Qu, Mat& Pu);
		Mat features(Mat& frame, HIST_TYPE type);
		Rect track(Mat& frame, HIST_TYPE type);
	};
}

#endif
