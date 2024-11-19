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

	SettingsTexture(const std::string& fileName);

private:
	void setMainClasses(const json& channelJson);
	void setClassesTexture();
	void repaintTextureImage(const std::string& textureName, std::map<std::string, int>& textureClasses);
};

