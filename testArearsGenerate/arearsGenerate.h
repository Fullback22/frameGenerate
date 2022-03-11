#include <random>
#include <iostream>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ProbabilityOfPosition.h"

class ArearsGenerate
{
	size_t quantityClases{};
	size_t quantityNotNullClasses{};
	std::vector<int> weigthsOnStep{};
	std::vector<int> weigthsInitial{};
	
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> subClassesMasks{};
	std::vector<cv::Mat> mainClassesMasks{};
	
	std::vector<std::vector<float>> weigthMap{};
	std::vector<std::vector<int>> classeMap{};

	std::vector<std::vector<int>> transitionMap{};

	std::vector<double> startPropobility{};
	ProbabilityOfPosition* propobilityOfPosition{ nullptr };
	std::vector<double> propobilityOfNeighbors{};

	int activNeighbors{};
	int quantityNeighbors{};

	std::random_device rd;
	std::mt19937 gen;

	void computeQuantityNeihbors();
	void setClasseMapSize();
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(std::vector<float>const *newWeigth);
	std::vector<float> computeFrequencyOfPosition(cv::Point const& activPoint);
	void computeExtensionWeigths(std::vector<float> const* classesWeigth);
	void computeNewWeigths(std::vector<float> const* classesWeigth);
	template <typename T>
	std::vector<float> fromWeigthToPropabilitys(std::vector<T> const* weigth);
	void initMatVector(std::vector<cv::Mat>& inputVector);
	void setClassesParametrs(std::vector<int> const* frequencyClasses,
							cv::Size const  newCalsSize,
							cv::Size const weigthMapSize,
							const std::vector<float>* weigthsForWeigthMap);

	
	void fromFrequencyToPropobility(std::vector<int> const* frequncy, std::vector<double>& propobility);
	void fromPropobilityToFrequency(std::vector<double> const* propobility, std::vector<int>& frequncy, int accurusy = 1000);
	void correctionPropobilityOfNeighbors(double const propobilityOfPosition, double& propobilityOfNeighbors);
	void generateClasseMap();
	void initClassesMasks(std::vector<cv::Mat>& classesMasks);
	void initMainImage();
	void combinateMainAndSubClasses(int numberMainClass);
public:
	ArearsGenerate(cv::Size const mainImageSize);
	void setPropobilityOfPosition(ProbabilityOfPosition const *newPropobilityOfPosition);
	void setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap);
	void setMainClassesParametrs(std::vector<int> const* frequencyClasses = new std::vector<int>{ 1, 1 },
							cv::Size const newCalsSize = cv::Size(1, 1),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 20,5,0,20,0,20,0,0 });

	void setSubClassesParametrs(std::vector<int> const* frequencyClasses = new std::vector<int>{ 1,1,1,1 },
							cv::Size const newCalsSize = cv::Size(8, 8),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 1,1,1,1,1,1,1,1 });
	
	cv::Mat* generateImageWithMainClasess();
	cv::Mat* generateImageWithSubClasess();
	int getNewValue(std::vector<double> &const propobility);
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
