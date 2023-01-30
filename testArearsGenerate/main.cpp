#include "ArearsGenerate.h"
#include "MySigmoid.h"
int main()
{
	cv::Size imageSize{ 600,400 };
	ArearsGenerate test{ imageSize };
	
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::vector<std::vector<double>> probabilityOfPosition(5);
	std::vector<std::vector<int>> transitionMap(5);
	transitionMap[0] = { 1,1,1,1,0 };
	transitionMap[1] = { 1,1,1,1,0 };
	transitionMap[2] = { 0,0,1,1,1 };
	transitionMap[3] = { 0,0,1,1,1 };
	transitionMap[4] = { 0,0,1,1,1 };
	
	std::uniform_int_distribution<> initDist{ 120, 170 };
	int imageNameOffset{ 0 };
	
	for (int i{ 0 }; i < 1; ++i)
	{
		double positionOffset{ static_cast<double>(initDist(gen)) };
		MySigmoid initProbabilityOfPositionMainClasses{ positionOffset, 0.3 };
		for (int j{ 0 }; j < imageSize.height; ++j)
		{
			probabilityOfPosition[0].push_back(initProbabilityOfPositionMainClasses.getValue(j) / 2);
			probabilityOfPosition[1].push_back(probabilityOfPosition[0][j]);
			probabilityOfPosition[2].push_back(0.5 - probabilityOfPosition[0][j]);
			probabilityOfPosition[3].push_back(probabilityOfPosition[2][j]);
			probabilityOfPosition[4].push_back(probabilityOfPosition[2][j]);
		}

		ProbabilityOfPosition probobility{ 20, 50, imageSize.width / 7, imageSize.width / 5, 0.5, 3 };
		probobility.setProbability(&probabilityOfPosition);
		test.setClassesParametrs();
		test.setProbabilityOfPosition(probobility);
		test.setTrasitionMap(transitionMap);

		cv::Mat imageWithMainClasses(test.generateImage());
		cv::imwrite("test/background" + std::to_string(i + imageNameOffset) + ".png", imageWithMainClasses);
		std::cout << i << std::endl;
		
		cv::imshow("test", imageWithMainClasses);


		cv::waitKey();
	}
	return 0;
}