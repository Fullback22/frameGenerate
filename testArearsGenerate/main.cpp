#include "ArearsGenerate.h"
#include "MySigmoid.h"

struct ModelParametr
{
	size_t quantityImage{ 0 };
	size_t startNumber{ 0 };
	size_t quantityClases{ 4 };
	cv::Size callSize{ 1,1 };
	cv::Size weigthMapSize{ 3, 3 };
	std::vector<double> weigthsForWeigthMap{ 0.3,1.0,0.3,1.0,1.0,0.3,1.0,0.3 };
	float landProportion{ 0.5 };
	std::vector<std::vector<int>> transitionMap;

};

void readModelParametr(const std::string& jsonFileName);

int main()
{
	size_t quantityImage{ 1 };
	size_t startNumber{ 0 };
	size_t quantityClases{ 4 };
	cv::Size callSize{ 1,1 };
	cv::Size weigthMapSize{ 3, 3 };
	std::vector<double> weigthsForWeigthMap{ 0.3,1.0,0.3,1.0,1.0,0.3,1.0,0.3 };
	float landProportion{ 0.5 };
	std::vector<std::vector<int>> transitionMap(quantityClases);
	transitionMap[0] = { 1,1,1,1 };
	transitionMap[1] = { 1,1,1,1 };
	transitionMap[2] = { 0,0,1,1 };
	transitionMap[3] = { 0,0,1,1 };
	transitionMap[4] = { 0,0,1,1,1 };

	std::vector<cv::Size> standartImageSize{ {640, 480}, {800,600}, {960, 540}, {1024, 600}, {1280, 720}, {1280, 1024}, {1600, 900}, {1920, 1080}, {2048,1080} };
	int quantityOfSize{ static_cast<int>(standartImageSize.size()) };

	std::uniform_int_distribution<> imageSizeDistr{ 0, quantityOfSize - 1 };
	std::random_device rd{};
	std::mt19937 generator{ rd() };


	for (int i{ 0 }; i < quantityImage; ++i)
	{
		int numberOfImageSize{ imageSizeDistr(generator) };
		cv::Size imageSize{ standartImageSize[numberOfImageSize] };
			
		int landAirBorder{  static_cast<int>(imageSize.height *landProportion) };
		int landAirSigma = landAirBorder * 0.1;
		//int forestBorder{ landAirBorder + landAirBorder / 4 };




		std::uniform_int_distribution<> initDist{ landAirBorder - landAirSigma, landAirBorder + landAirSigma };
		double positionOffset{ static_cast<double>(initDist(generator)) };


		std::vector<std::vector<double>> probabilityOfPosition(quantityClases, std::vector<double>(imageSize.height));
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
		myModel.setClassesParametrs(quantityClases, callSize, weigthMapSize, weigthsForWeigthMap);
		myModel.setProbabilityOfPosition(probobility);
		myModel.setTrasitionMap(transitionMap);

		cv::Mat imageWithMainClasses(myModel.generateImage());
		cv::imwrite("myModel_areas/myModel_" + std::to_string(i + startNumber) + ".png", imageWithMainClasses);
		std::cout << i + startNumber << std::endl;
		//cv::imshow("test", imageWithMainClasses);
		//cv::waitKey(23);
	}
	return 0;
}

void readModelParametr(const std::string& jsonFileName)
{


}
