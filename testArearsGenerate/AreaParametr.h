#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include "json.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "MySigmoid.h"

struct AreaParametr
{
	using json = nlohmann::json;

	size_t quantityImage{ 0 };
	size_t startNumber{ 0 };
	size_t quantityClasses{ 4 };
	cv::Size callSize{ 1,1 };
	cv::Size weigthMapSize{ 3, 3 };
	std::vector<double> weigthsForWeigthMap{ 0.3,1.0,0.3,1.0,1.0,0.3,1.0,0.3 };
	double landAirProportion{ 0.5 };
	std::vector<std::vector<int>> transitionMap;
	std::vector<std::string> probabilityOfPosition;
	int multiplicityResetToZeroOffset{ 3 };
	int lowerOffsetUpdateValue{ 7 };
	int upperOffsetUpdateValue{ 5 };
	int lowerOffsetValue{ 50 };
	int upperOffsetValue{ 70 };
	float probabilityOfOffset{ 0.2 };

	AreaParametr(const std::string& fileName);
	void setProbabilityOfPosition(std::vector<std::vector<double>>& outProbability, double const positionOffset);

private:
	void parsString(const std::string& input, std::vector<std::string>& output, const std::string& delimiter = " ");
};

