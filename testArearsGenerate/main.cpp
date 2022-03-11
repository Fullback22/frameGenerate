#include "ArearsGenerate.h"
#include "MySigmoid.h"
int main()
{
	std::vector<int> frenquncesClasses{ 1,1 };
	cv::Size imageSize{ 600,400 };
	ArearsGenerate test{ imageSize };
	
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::vector<std::vector<double>> probabilityOfPosition(2);
	std::vector<std::vector<int>> transitionMap(2);
	transitionMap[0] = { 1,1 };
	transitionMap[1] = { 0,1 };
	std::uniform_int_distribution<> initDist{ 120, 170 };
	MySigmoid initProbabilityOfPosition{ static_cast<double>(initDist(gen)), 0.3 };
	for (int j{ 0 }; j < imageSize.height; ++j)
	{
		probabilityOfPosition[0].push_back(initProbabilityOfPosition.getValue(j));
		probabilityOfPosition[1].push_back(1.0 - probabilityOfPosition[0][j]);
	}
	
	ProbabilityOfPosition probobility{ 20, 50, imageSize.width / 7, imageSize.width / 5, 0.5, 3 };
	probobility.setProbability(&probabilityOfPosition);
	test.setMainClassesParametrs();
	test.setPropobilityOfPosition(&probobility);
	test.setTrasitionMap(&transitionMap);

	cv::Mat imageWithMainClasses(*test.generateImageWithMainClasess());
	cv::imshow("test", imageWithMainClasses);
	cv::waitKey();
}