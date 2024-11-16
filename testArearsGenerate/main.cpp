#include "ArearsGenerate.h"

#include "AreaParametr.h"


#include <Windows.h>

void readAreaParametr(const std::string& jsonFileName);


int main()
{
	AreaParametr param{ "modelParametrs.json" };

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

		param.setProbabilityOfPosition(probabilityOfPosition, positionOffset);

		ProbabilityOfPosition probobility{ param.lowerOffsetValue, param.upperOffsetValue, imageSize.width / param.upperOffsetUpdateValue, imageSize.width / param.lowerOffsetUpdateValue, 0.2, param.multiplicityResetToZeroOffset };
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

void readAreaParametr(const std::string& jsonFileName)
{

}


