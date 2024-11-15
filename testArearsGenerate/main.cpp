#include "ArearsGenerate.h"
#include "MySigmoid.h"
#include "json.hpp"
#include <fstream>
#include <Windows.h>
#include <string>

struct ModelParametr
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

	ModelParametr(const std::string& fileName);
};

void readModelParametr(const std::string& jsonFileName);

void setProbabilityOfPosition(const ModelParametr& modelParametrs, std::vector<std::vector<double>>& outProbability, double const positionOffset);

void parsString(const std::string& input, std::vector<std::string>& output, const std::string& delimiter = " ");

int main()
{
	ModelParametr param{ "modelParametrs.json" };

	std::vector<cv::Size> standartImageSize{ {640, 480}, {800,600}, {960, 540}, {1024, 600}, {1280, 720}, {1280, 1024}, {1600, 900}, {1920, 1080}, {2048,1080} };
	int quantityOfSize{ static_cast<int>(standartImageSize.size()) };

	std::uniform_int_distribution<> imageSizeDistr{ 0, quantityOfSize - 1 };
	std::random_device rd{};
	std::mt19937 generator{ rd() };


	for (int i{ 0 }; i < param.quantityImage; ++i)
	{
		const int numberOfImageSize{ imageSizeDistr(generator) };
		const cv::Size& imageSize{ standartImageSize[numberOfImageSize] };
			
		int landAirBorder{ static_cast<int>(imageSize.height * param.landAirProportion) };
		int landAirSigma = landAirBorder * 0.1;

		std::uniform_int_distribution<> initDist{ landAirBorder - landAirSigma, landAirBorder + landAirSigma };
		double positionOffset{ static_cast<double>(initDist(generator)) };


		std::vector<std::vector<double>> probabilityOfPosition(param.quantityClasses, std::vector<double>(imageSize.height));

		setProbabilityOfPosition(param, probabilityOfPosition, positionOffset);

		ProbabilityOfPosition probobility{ 50, 80, imageSize.width / 7, imageSize.width / 5, 0.2, 3 };
		probobility.setProbability(probabilityOfPosition);

		ArearsGenerate myModel{ imageSize };
		myModel.setClassesParametrs(param.quantityClasses, param.callSize, param.weigthMapSize, param.weigthsForWeigthMap);
		myModel.setProbabilityOfPosition(probobility);
		myModel.setTrasitionMap(param.transitionMap);

		cv::Mat imageWithMainClasses(myModel.generateImage());

		cv::imwrite("myModel_areas/myModel_" + std::to_string(i + param.startNumber) + ".png", imageWithMainClasses);
		std::cout << i + param.startNumber << std::endl;
	}
	return 0;
}

void readModelParametr(const std::string& jsonFileName)
{

}

void setProbabilityOfPosition(const ModelParametr& modelParametrs, std::vector<std::vector<double>>& outProbability, double const positionOffset)
{
	MySigmoid initProbabilityOfPositionMainClasses{ positionOffset, 0.3 };
	std::vector<std::vector<std::string>> probabilityParametrs(modelParametrs.probabilityOfPosition.size());

	std::vector<std::string> probabilityBufer{ modelParametrs.probabilityOfPosition };
	for (size_t i{}; i < probabilityParametrs.size(); ++i)
	{
		if (atoi(probabilityBufer[i][0]) 
		parsString(probabilityBufer[i], probabilityParametrs[i]);

	}


	for (int j{ 0 }; j < outProbability[0].size(); ++j)
	{
		for (int i{}; i < modelParametrs.probabilityOfPosition.size(); ++i)
		{
			outProbability[0][j] = initProbabilityOfPositionMainClasses.getValue(j) / 2.0;
			outProbability[1][j] = outProbability[0][j];
			outProbability[2][j] = 0.5 - outProbability[0][j];
			outProbability[3][j] = outProbability[2][j];
		}
	}
}

void parsString(const std::string& input, std::vector<std::string>& output, const std::string& delimiter)
{
	size_t pos_start{};
	size_t pos_end{};
	size_t delim_len{ delimiter.size() };
	std::string token;

	while ((pos_end = input.find(delimiter, pos_start)) != std::string::npos) 
	{
		token = input.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		output.push_back(token);
	}

	output.push_back(input.substr(pos_start));
}

ModelParametr::ModelParametr(const std::string& fileName)
{
    if (fileName != "")
    {
        json modelParametrs;

        std::ifstream paramFile(fileName);

        if (!paramFile.is_open()) {
            std::cout << "ERROR: File not opened" << std::endl;
        }
        else
        {
            std::cout << "OK: File opened" << std::endl;

            try 
            {
                modelParametrs = json::parse(paramFile);
            }
            catch (...) 
            {
                std::cout << "ERROR: JSON OPEN" << std::endl;
                paramFile.close();
            }
            try
            {
                json channelJson = modelParametrs.at("MODEL_PARAMETRS");
				quantityImage = channelJson.at("quantityImage").get<size_t>();
				startNumber = channelJson.at("startNumber").get<size_t>();
				quantityClasses = channelJson.at("quantityClasses").get<size_t>();
				callSize.width = channelJson.at("callWidth").get<int>();
				callSize.height = channelJson.at("callHeigth").get<int>();
				weigthMapSize.width = channelJson.at("weigthMapWidth").get<int>();
				weigthMapSize.height = channelJson.at("weigthMapHeigth").get<int>();
				weigthsForWeigthMap = channelJson.at("weigthMap").get<std::vector<double>>();
				landAirProportion = channelJson.at("landAirProportion").get<double>();
				transitionMap.resize(quantityClasses);
				transitionMap = channelJson.at("transitionMap").get<std::vector<std::vector<int>>>();
				probabilityOfPosition = channelJson.at("probabilityOfPosition").get<std::vector<std::string>>();

            }
            catch (...) 
            {
                std::cout << "ERROR: JSON READ" << std::endl;
                paramFile.close();
            }
            paramFile.close();
        }
    }
}
