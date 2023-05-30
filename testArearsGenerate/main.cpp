#include "ArearsGenerate.h"
#include "MySigmoid.h"
int main()
{
	size_t quantityImage{ 1080 };
	size_t startNumber{ 0 };

	std::vector<int> imageWidth{ 640, 800, 960, 1024, 1280, 1280, 1600, 1920, 2048 };
	std::vector<int> imageHeigth{ 480, 600, 540, 600, 720, 1024, 900, 1080, 1080 };
	int quantityOfSize{ static_cast<int>(imageHeigth.size()) };

	std::uniform_int_distribution<> imageSizeDistr{ 0, quantityOfSize - 1 };

	std::random_device rd{};
	std::mt19937 generator{ rd() };
	for (int i{ 0 }; i < 10; ++i)
	{
		int imageNameOffset{ 0 };
		size_t quantityClases{ 5 };
		int numberOfImageSize{ imageSizeDistr(generator) };
		cv::Size imageSize{ imageWidth[numberOfImageSize], imageHeigth[numberOfImageSize] };
		cv::Size callSize{ 5,5 };
		cv::Size weigthMapSize{ 3, 3 };
		std::vector<double> weigthsForWeigthMap{ 0.5,1.0,0.5,1.0,1.0,0.2,0.3,0.2 };
		float landAirProportion{ 2 / 1 };
		int landAirBorder{};
		if (landAirProportion > 1)
			landAirBorder = imageSize.height / landAirProportion;
		else
			landAirBorder = imageSize.height * landAirProportion;
		int landAirSigma = landAirBorder * 0.1;
 		
		std::vector<std::vector<int>> transitionMap(quantityClases);
		transitionMap[0] = { 1,1,1,1,0 };
		transitionMap[1] = { 1,1,1,1,0 };
		transitionMap[2] = { 0,0,1,1,1 };
		transitionMap[3] = { 0,0,1,1,1 };
		transitionMap[4] = { 0,0,1,1,1 };


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
			if (j > 250)
			{
				probabilityOfPosition[2][j] = 0;
				probabilityOfPosition[3][j] = 0.5;
				probabilityOfPosition[4][j] = 0.5;
			}
		}

		ProbabilityOfPosition probobility{ 50, 80, imageSize.width / 7, imageSize.width / 5, 0.0, 3 };
		probobility.setProbability(probabilityOfPosition);

		ArearsGenerate myModel{ imageSize };
		myModel.setClassesParametrs(quantityClases, callSize, weigthMapSize, weigthsForWeigthMap);
		myModel.setProbabilityOfPosition(probobility);
		myModel.setTrasitionMap(transitionMap);

		cv::Mat imageWithMainClasses(myModel.generateImage());
		cv::imwrite("myModel_areas/myModel_" + std::to_string(i + imageNameOffset) + ".png", imageWithMainClasses);
		std::cout << i << std::endl;
		cv::imshow("test", imageWithMainClasses);

		cv::waitKey(23);
	}
	return 0;
}