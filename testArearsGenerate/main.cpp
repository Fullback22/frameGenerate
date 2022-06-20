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
	int imageNameOffset{ 0 };
	for (int i{ 0 }; i < 1; ++i)
	{
		double positionOffset{ static_cast<double>(initDist(gen)) };
		MySigmoid initProbabilityOfPositionMainClasses{ positionOffset, 0.3 };
		for (int j{ 0 }; j < imageSize.height; ++j)
		{
			probabilityOfPosition[0].push_back(initProbabilityOfPositionMainClasses.getValue(j));
			probabilityOfPosition[1].push_back(1.0 - probabilityOfPosition[0][j]);
		}

		ProbabilityOfPosition probobility{ 20, 50, imageSize.width / 7, imageSize.width / 5, 0.5, 3 };
		probobility.setProbability(&probabilityOfPosition);
		test.setMainClassesParametrs();
		test.setProbabilityOfPosition(&probobility);
		test.setTrasitionMap(&transitionMap);
		test.setWeigthProbabilitys(1.0, 1.0);

		cv::Mat imageWithMainClasses(test.generateImageWithMainClasess());
		cv::imwrite("test/background" + std::to_string(i + imageNameOffset) + ".png", imageWithMainClasses);
		std::cout << i << std::endl;
		cv::destroyWindow("test");
		cv::imshow("test", imageWithMainClasses);
		probabilityOfPosition.clear();
		probabilityOfPosition.assign(3, std::vector<double>());
		
		for (int j{ 0 }; j < imageSize.height; ++j)
		{
			double newProbabilityOfPosition{ initProbabilityOfPositionMainClasses.getValue(j) };
			if (newProbabilityOfPosition > 1.3)
			{
				probabilityOfPosition[0].push_back(initProbabilityOfPositionMainClasses.getValue(j));
				probabilityOfPosition[1].push_back((1.0 - probabilityOfPosition[0][j]) / 2.0);
				probabilityOfPosition[2].push_back((1.0 - probabilityOfPosition[0][j]) / 2.0);
			}
			else
			{
				probabilityOfPosition[0].push_back(0.3);
				probabilityOfPosition[1].push_back(0.3);
				probabilityOfPosition[2].push_back(0.3);
			}
		}

		probobility.setProbabilityOfOffset(0.0);
		probobility.setProbability(&probabilityOfPosition);
		test.setSubClassesParametrs();
		test.setProbabilityOfPosition(&probobility);
		test.setWeigthProbabilitys(1.0, 10.0);
		
		transitionMap[0] = { 1,1,1 };
		transitionMap[1] = { 1,1,1 };
		transitionMap.push_back({ 1, 1, 1 });
		
		test.setTrasitionMap(&transitionMap);

		cv::Mat imageWithSubClasses(test.generateImageWithSubClasess(1));
		cv::imshow("test1", imageWithSubClasses);


		cv::waitKey(22);
	}
}