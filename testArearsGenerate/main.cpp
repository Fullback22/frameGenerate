#include "arearsGenerate.h"
#include "MySigmoid.h"
int main()
{
	std::vector<int> frenquncesClasses{ 1,1 };
	cv::Size imageSize{ 600,400 };
	ArearsGenerate test{ imageSize };
	
	std::random_device rd1;
	std::mt19937 gen(rd1());

	std::vector<std::vector<double>> propobilityOfPosition(2);
	std::vector<std::vector<int>> transitionMap(2);
	transitionMap[0] = { 1,1 };
	transitionMap[1] = { 0,1 };
	std::uniform_int_distribution<> initDist{ 120, 170 };
	MySigmoid initPropobilityOfPosition{ static_cast<double>(initDist(gen)), 0.3 };
	for (int j{ 0 }; j < imageSize.height; ++j)
	{

		
		propobilityOfPosition[0].push_back(initPropobilityOfPosition.getValue(j));
		propobilityOfPosition[1].push_back(1.0 - propobilityOfPosition[0][j]);
	}
	

	test.setMainClassesParametrs();
	test.setPropobilityOfPosition(&propobilityOfPosition);
	test.setTrasitionMap(&transitionMap);
	test.generateMainClasseMap();
	test.initClassesMasks();
	test.initMainImage();
}