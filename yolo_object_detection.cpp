#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

const size_t network_width = 416;
const size_t network_height = 416;

int main(int argc, char** argv)
{
	String modelConfiguration = "yolo.cfg";
	String modelBinary =		"yolo.weights";
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    if (net.empty()) {
		return -1;
    }

	cv::Mat frame, resized, inputBlob, detectionMat;
	cv::VideoCapture cap("video.mp4");
	if (!cap.isOpened()) {
		return -1;
	}

	vector<string> class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
		"truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
		"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
		"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
		"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
		"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
		"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
		"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
		"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
		"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
		"teddy bear", "hair drier", "toothbrush" };

	while (true) {

		cap >> frame;

		if (frame.empty()) {
			return 1;
		}

		cv::resize(frame, resized, cv::Size(network_width, network_height));
		inputBlob = blobFromImage(resized, 1 / 255.F);		//Convert Mat to batch of images
		net.setInput(inputBlob, "data");					//set the network input
		detectionMat = net.forward("detection_out");		//compute output

		float confidenceThreshold = 0.25;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			const int probability_index = 5;
			const int probability_size = detectionMat.cols - probability_index;
			float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

			size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
			float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

			if (confidence > confidenceThreshold)
			{
				float x = detectionMat.at<float>(i, 0);
				float y = detectionMat.at<float>(i, 1);
				float width = detectionMat.at<float>(i, 2);
				float height = detectionMat.at<float>(i, 3);
				float xLeftBottom = (x - width / 2) * frame.cols;
				float yLeftBottom = (y - height / 2) * frame.rows;
				float xRightTop = (x + width / 2) * frame.cols;
				float yRightTop = (y + height / 2) * frame.rows;

				cv::Rect object(
					(int)xLeftBottom,
					(int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				cv::Rect title(
					(int)xLeftBottom,
					(int)yLeftBottom - 30,
					(int)(xRightTop - xLeftBottom),
					30);
								
				std::string object_name = "";
				if (objectClass < class_names.size()) {
					object_name = class_names[objectClass];
				}
				cv::rectangle(frame, title, Scalar(0, 255, 0), -1);
				cv::putText(frame, object_name, cv::Point((int)xLeftBottom, (int)yLeftBottom - 2), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0), 3);
				cv::rectangle(frame, object, Scalar(0, 255, 0), 2);
			}
		}

		imshow("detections", frame);
		waitKey(1);
	}

    return 0;
} // main