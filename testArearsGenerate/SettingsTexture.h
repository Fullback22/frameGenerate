#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <list>
#include <filesystem>

#include "json.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "TextureSynthesis.h"

namespace fs = std::filesystem;

class SettingsTexture
{
	using json = nlohmann::json;

	std::map<std::string, int> classes;
	std::map<std::string, cv::Mat> textureImage;
	cv::Size2i textureBlock{ 50, 50 };
	cv::Mat mapImage{};
	cv::Mat mapImageWithTexture{};
	unsigned int maskColor{ 255 };

private:
	std::string getRandomTexture(const std::string& textureName);
	void setMainClasses(const json& channelJson);
	void setClassesTexture();
	std::string getClassName(const int value) const;
	void getBoundingBoxAndSetMask(cv::Rect2i& boundingBox, const cv::Point2i& startPoint);
	void toWest(int& minX, int const replaceableColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points);
	void toEast(int& maxX, int const replaceableColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points);
	void setTexture(const cv::Mat& textureImage, const cv::Rect2i& boundingBox);

public:
	SettingsTexture(const std::string& fileName);
	static void repaintImage(cv::Mat& arearsMap, std::map<std::string, int>& classes, const int startColor);

	void setMapImage(const cv::Mat& image);
	void addTextureToMapImage();
	void getImageWithTexture(cv::Mat outImage) const;
	void saveMapWithTexture(const std::string& fileName);
};