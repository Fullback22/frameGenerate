#pragma once
#include <fstream>
#include <string>
#include <iostream>

#include "json.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "TextureSynthesis.h"

class SettingsTexture
{
	using json = nlohmann::json;

	std::map<std::string, int> classes;
	std::map<std::string, cv::Mat> textureImage;
	cv::Size2i testureBlock{ 50, 50 };
	cv::Mat mapImage{};

private:
	void setMainClasses(const json& channelJson);
	void setClassesTexture();
	


public:
	SettingsTexture(const std::string& fileName, const cv::Mat& image);
	static void repaintImage(cv::Mat& arearsMap, std::map<std::string, int>& classes, const int startColor);
};