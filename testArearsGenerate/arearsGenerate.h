#include <random>
#include <iostream>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ProbabilityOfPosition.h"

class ArearsGenerate
{
	size_t quantityClasses{};
	size_t quantityNotNullClasses{};

	std::vector<int> weigthsOnStep{};
	std::vector<int> weigthsInitial{};
	
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> classesMasks{};
	
	std::vector<std::vector<float>> weigthMap{};
	std::vector<std::vector<int>> classMap{};

	std::vector<std::vector<int>> transitionMap{};

	ProbabilityOfPosition* probabilityOfPosition{ nullptr };
	std::vector<double> probabilityOfNeighbors{};

	int activNeighbors{};
	int quantityNeighbors{};
	double weigthProbabilityOfPosition{ 1.0 };
	double weigthProbabilityOfNeighbors{ 3.0 };

	std::random_device rd;
	std::mt19937 gen;

	void computeQuantityNeihbors();
	void setClassMapSize();
	void updateClassMap(const cv::Size& oldCalsSize);
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(std::vector<float>const *newWeigth);
	std::vector<float> computeFrequencyOfPosition(cv::Point const& activPoint);
	void computeExtensionWeigths(std::vector<float> const* classesWeigth);
	void computeNewWeigths(std::vector<float> const* classesWeigth);
	template <typename T>
	std::vector<float> fromWeigthToProbabilitys(std::vector<T> const* weigth);
	void initMatVector(std::vector<cv::Mat>& inputVector);
	void setClassesParametrs(int quantityClasses,
							cv::Size const  newCalsSize,
							cv::Size const weigthMapSize,
							const std::vector<float>* weigthsForWeigthMap);

	
	void fromFrequencyToProbability(std::vector<int> const* frequncy, std::vector<double>& propobility);

	void correctionProbabilityOfNeighbors(double const propobilityOfPosition, double& propobilityOfNeighbors);
	void generateClasseMap(int const iter);
	void generateClasseMapInitializingPart();
	void generateClasseMapIterativePart(int const iter);
	void initClassesMasks(std::vector<cv::Mat>& classesMasks);
	void suppressionEmissions(cv::Mat &inOutputClassMap, int const medianFilterSize = 3);

	void initMainImage();
	cv::Mat drawClasses(std::vector<cv::Mat>* const maskClsses);
public:
	ArearsGenerate(cv::Size const mainImageSize);
	void setProbabilityOfPosition(ProbabilityOfPosition const *newPropobilityOfPosition);
	void setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap);
	void setClassesParametrs(int const quantityClasses = 2,
							cv::Size const newCalsSize = cv::Size(1, 1),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 1,5,1,5,5,1,5,1 });
	
	void setWeigthProbabilitys(double weigthOfPosition, double weigthOfNeighbors);
	cv::Mat generateImage();
	int getNewValue(std::vector<double> &const propobility);

};

template<typename T>
inline std::vector<float> ArearsGenerate::fromWeigthToProbabilitys(std::vector<T> const* weigth)
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
