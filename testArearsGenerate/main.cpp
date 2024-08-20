#include "ArearsGenerate.h"
#include "MySigmoid.h"
#include "json.hpp"
#include <fstream>

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
	std::vector<std::vector<double>> probabilityOfPosition;

	ModelParametr(const std::string& fileName);
};

void readModelParametr(const std::string& jsonFileName);

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


		std::vector<std::vector<double>> probabilityOfPosition__________(param.quantityClasses, std::vector<double>(imageSize.height));

		MySigmoid initProbabilityOfPositionMainClasses{ positionOffset, 0.3 };
		for (int j{ 0 }; j < imageSize.height; ++j)
		{
			probabilityOfPosition[0][j] = initProbabilityOfPositionMainClasses.getValue(j) / 2.0;
			probabilityOfPosition[1][j] = probabilityOfPosition[0][j];
			probabilityOfPosition[2][j] = 0.5 - probabilityOfPosition[0][j];
			probabilityOfPosition[3][j] = probabilityOfPosition[2][j];
			probabilityOfPosition[4][j] = probabilityOfPosition[2][j];
		}

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
