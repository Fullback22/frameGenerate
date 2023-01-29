#include "ArearsGenerate.h"
#include "MySigmoid.h"
int main()
{
	cv::Mat testImage(cv::Size(9, 3), CV_8UC1);
	
	cv::Point activPoint{};
	activPoint.x = static_cast<int>(ceil(testImage.cols / 2.0) - 1);
	activPoint.y = static_cast<int>(ceil(testImage.rows / 2.0) - 1);
	char color{ 0 };
	testImage.at<uchar>(activPoint) = color;
	++color;
	size_t step{ 1 };
		
	int iterator{ -1 };
	bool activCoordinateIs_Y{ true };

	cv::Rect imageRect{ 0, 0, testImage.cols - 0, testImage.rows - 0 };
	
	for (int numberUpdatePixels{ imageRect.area() - 1 } ; numberUpdatePixels > 0; )
	{
		bool isPointContainsInRect{};
		for (size_t i{}; i < step; ++i)
		{
			if (activCoordinateIs_Y)
				activPoint.y += iterator;
			else
				activPoint.x += iterator;
			
			isPointContainsInRect = imageRect.contains(activPoint);
			if (isPointContainsInRect)
			{
				testImage.at<uchar>(activPoint) = color;
				++color;
				--numberUpdatePixels;
			}
			else
			{
				i = step;
				if (activCoordinateIs_Y)
					activPoint.x += step * iterator * -1;
				else
					activPoint.y += (step + 1) * iterator;
			}
		}

		if (!isPointContainsInRect)
		{
			//if(activCoordinateIs_Y)
			++step;
			iterator *= -1;
		}
		else
		{
			if (activCoordinateIs_Y)
				iterator *= -1;
			else
				++step;
			activCoordinateIs_Y = !activCoordinateIs_Y;
		}
	}

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