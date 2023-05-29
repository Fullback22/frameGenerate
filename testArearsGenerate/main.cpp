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
		int numberOfImageSize{ imageSizeDistr(generator) };
		cv::Size imageSize{ imageWidth[numberOfImageSize], imageHeigth[numberOfImageSize] };

		ArearsGenerate test{ imageSize };

		std::vector<std::vector<double>> probabilityOfPosition(5, std::vector<double>(imageSize.height));
		std::vector<std::vector<int>> transitionMap(5);
		transitionMap[0] = { 1,1,1,1,0 };
		transitionMap[1] = { 1,1,1,1,0 };
		transitionMap[2] = { 0,0,1,1,1 };
		transitionMap[3] = { 0,0,1,1,1 };
		transitionMap[4] = { 0,0,1,1,1 };

		std::uniform_int_distribution<> initDist{ 120, 170 };
		int imageNameOffset{ 0 };



		double positionOffset{ static_cast<double>(initDist(generator)) };
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
		test.setClassesParametrs();
		test.setProbabilityOfPosition(probobility);
		test.setTrasitionMap(transitionMap);

		cv::Mat imageWithMainClasses(test.generateImage());
		cv::imwrite("myModel_areas/myModel_" + std::to_string(i + imageNameOffset) + ".png", imageWithMainClasses);
		std::cout << i << std::endl;
		cv::imshow("test", imageWithMainClasses);

		cv::waitKey(23);
	}
	return 0;
}