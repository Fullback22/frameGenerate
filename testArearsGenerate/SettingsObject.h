#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <random>
#include <algorithm>

#include "json.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

struct objectParams
{
	std::map <std::string, int> mainClasses;
	std::vector<std::string> landClasses{ "forest", "meadow" };
	std::vector <std::string> airClasses{ "sky", "cloud" };
	int maxQuantityLandObject{ 2 };
	int maxQuantityAirObject{ 0 };
	int minQuantityLandObject{ 1 };
	int minQuantityAirObject{ 0 };
	int maxObjectHeight{ 40 };
	int	minObjectHeight{ 20 };
};

class SettingsObject
{
	using json = nlohmann::json;

	objectParams params{};
	int quantityAirObject{};
	int quantityLandObject{};
	int objectHeigth{ 10 };
	std::vector<cv::Mat> airImage;
	std::vector<cv::Mat> landImage;

	void setMainClasses(const json& channelJson);
	void loadImage();
	void resizeImage(cv::Mat& image);
	void updateImageHeigth();
	cv::Point getStartPopintForObject(const cv::Mat backgroundImage, const cv::Mat object);
public:
	SettingsObject(const std::string& fileName);
	void updateQuantityObjectOnImage();
	void setObject(cv::Mat& inOutImage);
};

