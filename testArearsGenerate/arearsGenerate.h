#include <random>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ArearsGenerate
{
	size_t quantityClases{};
	size_t quantityNotNullClasses{};
	std::vector<float> weigthsOnStep{};
	std::vector<float> weigthsInitial{};
	
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> subClassesMasks{};
	std::vector<cv::Mat> mainClassesMasks{};
	
	std::vector<std::vector<float>> weigthMap{};
	std::vector<std::vector<int>> classeMap{};

	std::random_device rd;
	std::mt19937 gen;

	void setClasseMapSize();
	//void initClasseMap();
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(std::vector<float>const *newWeigth);
	std::vector<float> computeClassesWeigth(cv::Point const& activPoint);
	void computeExtensionWeigths(std::vector<float> const* classesWeigth);
	void computeNewWeigths(std::vector<float> const* classesWeigth);
	std::vector<int> convertPropabilitysOnStepToInt(int const accuracy = 1000);
	template <typename T>
	std::vector<float> fromWeigthToPropabilitys(std::vector<T> const* weigth);
	void initMatVector(std::vector<cv::Mat>& inputVector);
	void setClassesParametrs(std::vector<int> const* frequencyClasses,
							cv::Size const  newCalsSize,
							cv::Size const weigthMapSize,
							const std::vector<float>* weigthsForWeigthMap);

	
public:
	ArearsGenerate(cv::Size const mainImageSize);
	void setMainClassesParametrs(std::vector<int> const* frequencyClasses = new std::vector<int>{ 1, 2 },
							cv::Size const newCalsSize = cv::Size(16,16),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 1,1,0,1,0,1,0,0 });

	void setSubClassesParametrs(std::vector<int> const* frequencyClasses = new std::vector<int>{ 1,1,1,1 },
							cv::Size const newCalsSize = cv::Size(8, 8),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 1,1,1,1,1,1,1,1 });
	

	void generateClasseMap();
	void initClassesMasks();
	void initMainImage();
};

template<typename T>
inline std::vector<float> ArearsGenerate::fromWeigthToPropabilitys(std::vector<T> const* weigth)
{
	T sumWeigth{ };
	for (auto& weigth : *weigth)
	{
		sumWeigth += weigth;
	}
	double averagePropability{ 1.0 / sumWeigth };
	std::vector<float> outPropabitys{};
	for (auto& weigth : *weigth)
	{
		outPropabitys.push_back(averagePropability * weigth);
	}
	return outPropabitys;
}
