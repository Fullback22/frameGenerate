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

	std::vector<double> weigthsOnStep{};
	std::vector<int> weigthsInitial{};
	
	cv::Size calsSize{};
	cv::Mat mainImage{};
	
	std::vector<std::vector<double>> weigthMap{};
	std::vector<std::vector<int>> classMap{};

	std::vector<std::vector<int>> transitionMap{};

	ProbabilityOfPosition* probabilityOfPosition{ nullptr };

	int activNeighbors{};
	int quantityNeighbors{};

	std::random_device rd;
	std::mt19937 gen;

	template <typename T>
	std::vector<double> fromWeigthToProbabilitys(const std::vector<T>& weigth);
	
	unsigned int getTransitionCoefficient(const cv::Point& activPoint, unsigned int const targetClass);
	void computeQuantityNeihbors();
	void setClassMapSize();
	void updateClassMap(const cv::Size& oldCalsSize);
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(std::vector<float>const *newWeigth);
	void computeWeigthFromPosition(const cv::Point& activPoint);
	void computeExtensionWeigths(std::vector<float> const* classesWeigth);
	void computeNewWeigths(std::vector<float> const* classesWeigth);
	
	
	void initMatVector(std::vector<cv::Mat>& inputVector);
	
	

	double correctionCoefficientOfNeighbors(double const coefficientOfPosition, double const coefficientOfNeighbors);
	void generateClasseMap(size_t const iter);
	void initClassesMasks(std::vector<cv::Mat>& classesMasks);

	cv::Mat drawClasses(std::vector<cv::Mat>* const maskClsses);
public:
	ArearsGenerate(cv::Size const mainImageSize);
	void setProbabilityOfPosition(ProbabilityOfPosition const *newPropobilityOfPosition);
	void setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap);
	void setClassesParametrs(int const quantityClasses = 2,
							cv::Size const newCalsSize = cv::Size(40, 40),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{ 1,5,1,5,5,1,5,1 });
	
	cv::Mat generateImage();
	int getNewValue(std::vector<double> &const propobility);

};

template<typename T>
inline std::vector<double> ArearsGenerate::fromWeigthToProbabilitys(const std::vector<T>& weigth)
{
	T sumWeigth{ };
	for (auto& weigth : weigth)
	{
		sumWeigth += weigth;
	}
	if (sumWeigth == 0.0)
	{
		double averagePropability{ 1.0 / weigth.size()};
		std::vector<double> outPropabitys(weigth.size(), averagePropability);
		return outPropabitys;
	}
	else
	{
		double averagePropability{ 1.0 / sumWeigth };
		std::vector<double> outPropabitys{};
		for (auto& weigth : weigth)
		{
			outPropabitys.push_back(averagePropability * weigth);
		}
		return outPropabitys;
	}
}
