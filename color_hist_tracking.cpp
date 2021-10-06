#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include "ShowManyImages.hpp"					//for displaying multiple images together
#include "tracker.hpp"

//namespaces
using namespace cv;
using namespace std;
using namespace tracker;

//main function
int main(int argc, char ** argv)
{
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "/home/nwonknu/Documents/AVSA/Lab_4/datasets";									//dataset location.
	std::string output_path = "./outvideos/";									//location to save output videos

	// dataset paths
	std::string sequences[] = {"bolt1",										//test data for lab4.1, 4.3 & 4.5
							   /*"sphere","car1",								//test data for lab4.2
							   "ball2","basketball",					//test data for lab4.4
							   "bag","ball","road",*/};						//test data for lab4.6
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	FusionTrack ft;															//color tracker object

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes
		HIST_TYPE type = HUE;

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;
		ft.resetTracker();
		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code
//			list_bbox_est.push_back(Rect(100,100,40,50));//we use a fixed value only for this demo program. Remove this line when you use your code
			//...
			// If model not initialized set it to ground truth of first frame
			if(!ft.modelInitialized()){
				int n_candidates1 = 9, n_candidates2 = 9;
				int delta1 = 6, delta2 = 5;
				int bins1 = 16, bins2 = 20;
				bool colorTrack = true;
				bool gradientTrack = true;
				ft.setModel(frame, list_bbox_gt[frame_idx-1], n_candidates1, delta1,
						bins1, n_candidates2, delta2, bins2, colorTrack, gradientTrack
						);
				list_bbox_est.push_back(list_bbox_gt[frame_idx-1]);
			}
			else{
				list_bbox_est.push_back(ft.track(frame, type));
			}
			//...
			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			//show & save data
			putText(frame, "Groundtruth", cv::Point(10,30),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
			putText(frame, "Estimation", cv::Point(10,45),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
			Mat roi = ft.getModelROI();
			ShowManyImages("Tracking for "+sequences[s], 2, frame, roi);
			outputvideo.write(frame);//save frame to output video

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
