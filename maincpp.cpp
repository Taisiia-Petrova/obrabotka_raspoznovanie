#include<windows.h>
//#include <sys/stat.h>
//#include <filesystem>
#include <iostream>
//#include <sstream>
#include <fstream>
//#include <iterator>
#include <string> 
//#include <algorithm>
#include "opencv2/highgui.hpp"
#include "opencv2/highgui.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <bitset>
#include <math.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/core/utility.hpp"

using namespace std;
using namespace cv;

struct tokens : std::ctype<char> {
	tokens() : std::ctype<char>(get_table()) {}

	static std::ctype_base::mask const* get_table() {
		typedef std::ctype<char> cctype;
		static const cctype::mask *const_rc = cctype::classic_table();

		static cctype::mask rc[cctype::table_size];
		std::memcpy(rc, const_rc, cctype::table_size * sizeof(cctype::mask));

		rc[','] = std::ctype_base::space;
		rc[' '] = std::ctype_base::space;
		return &rc[0];
	}
};

std::vector<int> splitToInt(const std::string& input) {
	static std::locale parse_locale = std::locale(std::locale(), new tokens());
	std::vector<int> output{};
	std::stringstream ss{ input };
	ss.imbue(parse_locale);
	std::istream_iterator<std::string> begin(ss);
	std::istream_iterator<std::string> end;
	std::vector<std::string> vstrings(begin, end);
	std::transform(vstrings.begin(), vstrings.end(), std::back_inserter(output), [](std::string& value) { return std::stoi(value); });
	return output;
}

/*static void help(char** argv)
{
	cout << "\nThis is a demo program shows how perspective transformation applied on an image, \n"
		"Using OpenCV version " << CV_VERSION << endl;

	cout << "\nUsage:\n" << argv[0] << " [image_name -- Default ../data/right.jpg]\n" << endl;

	cout << "\nHot keys: \n"
		"\tESC, q - quit the program\n"
		"\tr - change order of points to rotate transformation\n"
		"\tc - delete selected points\n"
		"\ti - change order of points to invers transformation \n"
		"\nUse your mouse to select a point and move it to see transformation changes" << endl;
}*/

static void onMouse(int event, int x, int y, int, void*);
Mat warping(Mat image, Size warped_image_size, vector< Point2f> srcPoints, vector< Point2f> dstPoints);

String windowTitle = "Perspective Transformation Demo";
String labels[4] = { "TL","TR","BR","BL" };
struct ProgramData {
	vector< Point2f> roi = {};
	int roiIndex = 0;
	bool dragging = false;
	int selected_corner_index = 0;
	bool endProgram = false;
};

void drawMarker(Mat& image, Ptr<aruco::Dictionary> dict, int marker, int size, Point2f pos) {
	Mat markerImage;
	aruco::drawMarker(dict, marker, size, markerImage);
	int x = std::floor(pos.x);
	int y = std::floor(pos.y);
	markerImage.copyTo(image.colRange(pos.x, pos.x + size).rowRange(pos.y, pos.y + size));
}

std::vector<Point2f> generateMarkerPositions(int multiplier, int correction = 0) {
	float width = 1297, height = 910;
	vector<Point2f> corners(4);
	corners[0].x = 5;
	corners[0].y = 5;
	corners[1].x = width - correction-5;
	corners[1].y = 5;
	corners[2].x = width - correction-5;
	corners[2].y = height - correction-5;
	corners[3].x = 5;
	corners[3].y = height - correction-5;
	return corners;
}

bool extract(Mat& image, vector<Point2f>& roi, Mat& result) {
	float width = 1297, height = 910;
	if (roi.size() == 4) {
		Size result_size = Size(width, height);
		Mat H = findHomography(roi, generateMarkerPositions(1)); 
		warpPerspective(image, result, H, result_size); // do perspective transformation
		return true;
	}
	return false;
}

std::vector<Point2f> calculateRoi(Mat& image, Ptr<aruco::Dictionary> dictionary) {
	std::vector<int> markerIds;
	std::vector<std::vector<Point2f>> markerCorners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	parameters->minMarkerPerimeterRate = 0.1;
	aruco::detectMarkers(image, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

	vector<Point2f> roi(4);
	std::bitset<4> checksum;
	for (int index = 0; index < markerIds.size(); index++) {
		if (markerIds.at(index) > 45) {
			int part = 49 - markerIds.at(index);
			if (!checksum.test(part)) {
				roi[part] = markerCorners.at(index)[part];
				checksum.flip(part);
			}
		}
	}
	return checksum.all() ? roi : std::vector<Point2f>(0);
}

Mat drawMarkers(Mat original, vector<Point2f>& roi) {
	Mat image = original.clone();

	for (size_t i = 0; i < roi.size(); ++i) {
		line(image, roi[i], roi[(i + 1) % roi.size()], Scalar(0, 0, 255), 2);
	}
	for (size_t i = 0; i < roi.size(); ++i) {
		circle(image, roi[i], 5, Scalar(0, 255, 0), 3);
		putText(image, labels[i].c_str(), roi[i], QT_FONT_NORMAL, 0.8, Scalar(255, 0, 0), 2);
	}

	return image;
}

void processKeypress(ProgramData& data) {
	char c = (char)waitKey(10);

	if ((c == 'q') | (c == 'Q') | (c == 27)) {
		data.endProgram = true;
	}

	if ((c == 'c') | (c == 'C')) {
		data.roi.clear();
	}

	if ((c == 'r') | (c == 'R')) {
		data.roi.push_back(data.roi[0]);
		data.roi.erase(data.roi.begin());
	}

	if ((c == 'i') | (c == 'I')) {
		swap(data.roi[0], data.roi[1]);
		swap(data.roi[2], data.roi[3]);
	}
}

void processImage(Mat& image, Mat& extracted, Ptr<aruco::Dictionary> dictionary) {
	vector<Point2f> roi = calculateRoi(image, dictionary);
	imshow("Original", drawMarkers(image, roi));
	if (extract(image, roi, extracted)) {
		imshow("Scan", extracted);
	}
}


int main(int argc, char** argv)
{
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
	CommandLineParser parser(argc, argv, "{generate g||}{camera c||}{input i|test.jpg|}{output o|blank.jpg|}");
	vector<vector<vector<Rect2f>>> vecCoord(11);
	vector<Rect2f> vecRasd(11);
	vector<vector<pair<Point2f,Point2f>>> vecLines(11);
	vector<Rect2f> IDN(1);
	vector<vector<Rect2f>> ID(10);
	vector<pair<Point2f, Point2f>> IDL(10);
	int l,q,a;
    ifstream in("D:/DIP/Project4/Project4/test/in.txt"); // окрываем файл для чтения
    if (in.is_open())
    {
		in >> l;
		in >> q;
		in >> a;
    }
    in.close();     // закрываем файл

	int x = 200;
	int y = 90;
	int x_2 = 200;
	int width = 1297, height = 910, multiplier = 4;
	cv::Mat inputImage;
	cv::Mat extractedImage;
	cv::Point pt1(55, 90);
	cv::Point pt2(55 + width / 10, 90 + height / 2.5);
	IDN[0] = Rect2f(pt1, pt2);
	int y1 = 90 + height / 25;
	for (int r = 0; r < 10; r++)
	{
		cv::Point pt1(55, y1);
		cv::Point pt2(55 + width / 10, y1);
		IDL[r] = make_pair(pt1,pt2);
		ID[r].resize(3);
		int x1 = 55;
		for (int k = 0; k < 3; k++)
		{
			cv::Point pt1(x1, y1 - 30);
			cv::Point pt2(x1 + width / 40, y1 - 3);
			ID[r][k] = Rect2f(pt1, pt2);
			x1 = x1 + width / 30;
		}
		y1 = y1 + height / 25;
	}
	int y_2 = y + height / 2.5 + 40;
	for (int t = 0; t < 6; t++)
	{
		cv::Point pt1(x, y);
		cv::Point pt2(x + width / 8, y + height / 2.5);
		vecCoord[t].resize(16);
		vecRasd[t] = Rect2f(pt1, pt2);
		vecLines[t].resize(16);
		int y_1 = 87 + height / 38;
		for (int r = 0; r < 16; r++)
		{
			cv::Point pt1(x, y_1);
			cv::Point pt2(x + width / 8, y_1);
			vecCoord[t][r].resize(5);
			vecLines[t][r] = make_pair(pt1, pt2);
			int x_1 = x + width / 49;
			for (int k = 0; k < 5; k++)
			{
				cv::Point pt1(x_1, y_1 - 18);
				cv::Point pt2(x_1 + width / 57, y_1 - 2);
				vecCoord[t][r][k] = Rect2f(pt1, pt2);
				x_1 = x_1 + width / 48;
			}
			y_1 = y_1 + height / 38;
		}
		x = x + width / 8 + 10;
	}
	for (int t = 6; t < 11; t++)
	{
		cv::Point pt1(x_2, y_2);
		cv::Point pt2(x_2 + width / 8, y_2 + height / 2.5);
		vecCoord[t].resize(16);
		vecRasd[t] = Rect2f(pt1, pt2);
		vecLines[t].resize(16);
		int y_3 = y_2 + height / 38;
		for (int r = 0; r < 16; r++)
		{
			cv::Point pt1(x_2, y_3);
			cv::Point pt2(x_2 + width / 8, y_3);
			vecCoord[t][r].resize(5);
			vecLines[t][r] = make_pair(pt1, pt2);
			int x_1 = x_2 + width / 49;
			for (int k = 0; k < 5; k++)
			{
				cv::Point pt1(x_1, y_3 - 18);
				cv::Point pt2(x_1 + width / 57, y_3 - 2);
				vecCoord[t][r][k] = Rect2f(pt1, pt2);

				x_1 = x_1 + width / 48;
			}
			y_3 = y_3 + height / 38;
		}
		x_2 = x_2 + width / 8 + 10;
	}
	//
	//
	//
	int f;
	std::cin >> f;
	/*
	if (f == 1) {
		VideoCapture cap(0);
		if (!cap.isOpened()) {
			cout << "Error opening video stream or file" << endl;
			return -1;
		}
		ProgramData data = ProgramData();
		//namedWindow("Original", WINDOW_AUTOSIZE);
		namedWindow("Scan", WINDOW_AUTOSIZE);
		moveWindow("Scan", 20, 20);
		//moveWindow("Original", 330, 20);
		setMouseCallback(windowTitle, onMouse, &data);
		while (!data.endProgram)
		{
			cap >> inputImage;
			processImage(inputImage, extractedImage, dictionary);
			processKeypress(data);
		}
		return 0;
	} */
	if (f == 0) {
		Mat image = Mat::ones(height, width, CV_8UC1) * 255;
		int markerSize = multiplier * 8.5;
		std::vector<Point2f> corners = generateMarkerPositions(1, markerSize);
		for (int index = 0; index < corners.size(); index++) {
			drawMarker(image, dictionary, 49 - index, markerSize, corners.at(index));
		}
		cv::rectangle(image, IDN[0], cv::Scalar(250, 255,250));
		cv::putText(image, "ID number", cv::Point(IDN[0].x, IDN[0].y-2), FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(200, 200, 250), 1);
		for (int r = 0; r < 10; r++)
		{
			for (int k = 0; k < 3; k++)
			{
				cv::rectangle(image,ID[r][k],cv::Scalar(0, 255, 0));
				std::string s = std::to_string(r);
				cv::putText(image, s, cv::Point(ID[r][k].x+10, ID[r][k].y+18), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(200, 200, 250), 1);
			}
		}
		for (int t = 0; t < 6; t++) 
		{
			cv::rectangle(image, vecRasd[t], cv::Scalar(250, 250, 250));
			std::string s = std::to_string(t+1);
			cv::putText(image, "Subtest  " + s, cv::Point(vecRasd[t].x + 25, vecRasd[t].y -2), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1);

			for (int r = 0; r < 16; r++)
			{
				cv::line(image, vecLines[t][r].first, vecLines[t][r].second, cv::Scalar(250, 250, 250));
				std::string s = std::to_string(r + 1);
				cv::putText(image, s, cv::Point(vecLines[t][r].first.x+1, vecLines[t][r].first.y - 3), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1);
				for (int k = 0; k < 5;k++)
				{
					cv::rectangle(image, vecCoord[t][r][k], cv::Scalar(0, 255, 0));
				}
			}
		}
		for (int t = 6; t < 11; t++)
		{
			cv::rectangle(image, vecRasd[t], cv::Scalar(250, 255, 250));
			std::string s = std::to_string(t+1);
			cv::putText(image, "Subtest  " + s, cv::Point(vecRasd[t].x + 25, vecRasd[t].y-2), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1);
			for (int r = 0; r < 16; r++)
			{
				std::string s = std::to_string(r+1);
				cv::putText(image,s, cv::Point(vecLines[t][r].first.x+1, vecLines[t][r].first.y-3),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 200, 250), 1);
				cv::line(image, vecLines[t][r].first, vecLines[t][r].second, cv::Scalar(250, 255, 250));
				for (int k = 0; k < 5; k++)
				{
					cv::rectangle(image, vecCoord[t][r][k], cv::Scalar(0, 255, 0));
				}
			}
		}
		imshow("Blank", image);
		imwrite(parser.get<std::string>("output"), image);
		cv::waitKey();
		return 0;
	}

	std::ofstream out;          // поток для записи
	out.open("D:/DIP/Project4/Project4/test/result.txt", std::ios_base::app); // окрываем файл для записи
	WIN32_FIND_DATA FindFileData;
	HANDLE hf;
	hf = FindFirstFile("D:/DIP/Project4/Project4/test/*.jpg", &FindFileData);
	if (hf != INVALID_HANDLE_VALUE) {
		do {
			string name = FindFileData.cFileName;
			string n = "D:/DIP/Project4/Project4/test/" + name;
			Mat img = imread(n,0);
			processImage(img, extractedImage, dictionary);
			threshold(extractedImage, extractedImage, 200, 210, CV_THRESH_BINARY);
			cv::blur(extractedImage, extractedImage, Size(1, 1));
			imshow("Scan", extractedImage);
		vector<vector<int>> res (11);
		vector<int> num(3);
		for (int t = 0; t < 10; t++) {
			for (int r = 0; r < 3; r++)
			{
					Mat roi(extractedImage, ID[t][r]);
					Scalar m = mean(roi);
					if (m[0] < 150) {
						num[r] = t;
					}
		
			}
		}
		if (out.is_open()) out << "ID number " << num[0]<<num[1]<<num[2]<<endl;
		for (int t = 0; t < 11; t++) {
			out << "Subtest "<< t+1 << endl;
			res[t].resize(16);
			for (int r = 0; r < 16; r++) 
			{
					for(int k=0;k<5;k++)
					{
						Mat roi(extractedImage, vecCoord[t][r][k]);
						Scalar m = mean(roi);
						if (m[0] < 100) {
							if (res[t][r] == 0 || res[t][r] == -1) {
								res[t][r] = k + 1;
								out << "Question " << r + 1 << ":  " << res[t][r] << endl;
							}
							else out << "Other answer to the question " << r+1 << ":  " << k+1 << endl;
						}
						else if (res[t][r] == 0) res[t][r] = 0;				
					}
			}
		}
		int corr=0,corr1=0,corr2=0,corr3=0,corr4=0,corr5=0,corr6=0,corr7=0,corr8=0,corr9=0,corr10=0;
		int group = 0;
		int rawScore=0;
		std::vector<int> correct_answers1 = { splitToInt(" 5,1,5,4,1,1,1,1,3,4,5,5,4,1,1,2")};
		std::vector<int> correct_answers2 = { splitToInt(" 2,1,5,4,1,1,4,1,3,4,5,5,1,1,3,2")};
		std::vector<int> correct_answers3 = { splitToInt(" 5,1,3,4,1,3,1,1,3,4,5,5,1,2,1,2") };
		std::vector<int> correct_answers4 = { splitToInt(" 5,1,5,4,2,1,1,1,3,4,5,5,1,1,5,2") };
		std::vector<int> correct_answers5 = { splitToInt(" 4,1,5,4,5,1,5,1,3,4,5,5,1,3,1,2") };
		std::vector<int> correct_answers6 = { splitToInt(" 4,1,5,4,1,2,1,4,3,4,5,5,1,4,1,2") };
		std::vector<int> correct_answers7 = { splitToInt(" 5,1,1,4,1,1,1,1,3,4,5,5,2,1,1,2") };
		std::vector<int> correct_answers8 = { splitToInt(" 5,1,2,4,1,1,3,1,2,4,5,5,1,4,1,2") };
		std::vector<int> correct_answers9 = { splitToInt(" 2,4,5,4,1,5,1,3,5,4,5,5,1,1,5,2") };
		std::vector<int> correct_answers10 = { splitToInt(" 5,1,3,4,2,1,2,1,3,4,5,5,1,2,1,2") };
		std::vector<int> correct_answers11= { splitToInt(" 3,1,5,4,1,3,1,4,3,4,5,5,1,1,3,2") };

		std::vector<std::vector<int>> correct_answers_to_points = {
			splitToInt(" 2,4,6,7,9,10,11,12,13,14,15,15,16,17,19,20,20"),
			splitToInt(" 2,4,6,7,8, 9,10,11,12,13,13,14,15,16,17,18,19"),
			splitToInt(" 2,4,5,6,7, 8, 9,10,11,12,13,14,15,16,17,18,19"),
			splitToInt(" 2,4,5,7,8,10,11,12,13,14,15,15,16,17,18,19,20"),
			splitToInt(" 1,3,4,5,6, 8, 9,10,11,12,13,14,15,17,19,20,20"),
			splitToInt(" 2,3,5,6,7, 8,10,11,12,13,14,16,17,18,20,20,20"),
			splitToInt(" 2,4,6,8,9,10,10,11,12,13,14,15,16,17,18,19,20"),
			splitToInt(" 3,5,6,7,8, 9,10,10,11,12,13,14,15,17,18,19,20"),
			splitToInt(" 3,5,7,8,9,10,11,12,12,13,14,14,15,17,18,19,20"),
			splitToInt(" 2,4,5,6,7, 8, 9,10,11,12,13,14,15,17,19,20,20"),
			splitToInt(" 3,5,6,7,8, 8, 9,10,11,11,12,12,13,13,14,15,17")
		};
		std::vector<int> resultScore = splitToInt("             61,62,62,62,63,        63,64,64,64,65,"
			"     65,66,66,67,67,        68,68,68,69,69,        70,71,72,73,74,        75,76,77,78,79,        80,81,81,82,83,"
			"     83,84,85,85,86,        87,87,88,89,90,        90,91,92,92,93,        94,94,95,96,97,       97,98,99,99,100,"
			"101,101,102,103,103,   104,105,106,106,107,   108,108,109,110,110,   111,112,113,113,114,   114,115,116,117,118,"
			"118,119,120,120,121,   122,122,123,124,124,   125,126,127,128,129,   129,130,131,132,132,   133,134,134,135,136,"
			"137,138,138,139,139,   139,140,140,140,141,   141,141,142,142,142,   143,143,143,144,144,   144,145,145,145,146,"
			"146,146,147,147,147,   147,148,148,148,149,   149,149,150,150,150");
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers1[t]) corr ++;
		}
		rawScore = correct_answers_to_points.at(group).at(corr);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers2[t]) corr1++;
			//else ans[t] = 0;
		}	
		rawScore += correct_answers_to_points.at(group).at(corr1);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers3[t]) corr2++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr2);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers4[t]) corr3++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr3);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers5[t]) corr4++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr4);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers6[t]) corr5++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr5);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers7[t]) corr6++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr6);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers8[t]) corr7++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr7);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers9[t]) corr8++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr8);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers10[t]) corr9++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr9);
		group++;
		for (int t = 0; t < 16; t++) {
			if (res[group][t] == correct_answers11[t]) corr10++;
			//else ans[t] = 0;
		}
		rawScore += correct_answers_to_points.at(group).at(corr10);
		int r = resultScore.at(rawScore);
		out << endl<< "IQ: " << r << endl;
		out << endl;
				} while (FindNextFile(hf, &FindFileData) != 0);
				FindClose(hf);
		}
		out.close();
		ProgramData data = ProgramData();
		while (!data.endProgram) {
			processKeypress(data);
		}
		return 0;
	}
/*
static void onMouse(int event, int x, int y, int, void* genericData)
{
	ProgramData& data = *static_cast<ProgramData*>(genericData);
	// Action when left button is pressed
	if (data.roi.size() == 4) {
		for (int i = 0; i < 4; ++i) {
			if ((event == EVENT_LBUTTONDOWN) & ((abs(data.roi[i].x - x) < 10)) & (abs(data.roi[i].y - y) < 10)) {
				data.selected_corner_index = i;
				data.dragging = true;
			}
		}
	}
	else if (event == EVENT_LBUTTONDOWN) {
		data.roi.push_back(Point2f((float)x, (float)y));
	}

	// Action when left button is released
	if (event == EVENT_LBUTTONUP) {
		data.dragging = false;
	}

	// Action when left button is pressed and mouse has moved over the window
	if ((event == EVENT_MOUSEMOVE) && data.dragging) {
		data.roi[data.selected_corner_index].x = (float)x;
		data.roi[data.selected_corner_index].y = (float)y;
	}
}*/