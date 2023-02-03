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
	size_t activNeighbors{};
	size_t quantityNeighbors{};

	std::vector<std::vector<double>> weigthMap{};
	
	cv::Size calsSize{};
	std::vector<std::vector<int>> classMap{};
	
	std::vector<std::vector<int>> transitionMap{};
	std::vector<std::vector<std::vector<double>>> probabilityOfYMap{};
	ProbabilityOfPosition* probabilityOfPosition{ nullptr };
	std::vector<double> weigthsOnStep{};
	
	std::random_device randomDevice;
	std::mt19937 numberGenerator;
	
	cv::Mat mainImage{};

	
	template <typename T>
	std::vector<double> fromWeigthToProbabilitys(const std::vector<T>& weigth);
	
	unsigned int getTransitionCoefficient(const cv::Point& activPoint, unsigned int const targetClass);
	void computeQuantityNeihbors();
	void setClassMapSize();
	void updateClassMap(const cv::Size& oldCalsSize);
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(const std::vector<double>& newWeigth);
	void initProbabilityOfYMap();
	void computeWeigthFromPosition(const cv::Point& activPoint);

	
	
	void initMatVector(std::vector<cv::Mat>& inputVector);
	
	

	double correctionCoefficientOfNeighbors(double const coefficientOfPosition, double const coefficientOfNeighbors);
	double correctionCoefficientOf_Y(double const coefficientOfPosition, double const coefficientOfNeighbors);
	void generateClasseMap(size_t const iter);
	void initClassesMasks(std::vector<cv::Mat>& classesMasks);

	cv::Mat drawClasses(std::vector<cv::Mat>* const maskClsses);	
	int getNewValue(std::vector<double> &const propobility);
	void computeNewClassInPosition(const cv::Point& position, std::vector<std::vector<int>>* updatedClassMap = nullptr);
public:
	ArearsGenerate(cv::Size const mainImageSize);
	void setProbabilityOfPosition(const ProbabilityOfPosition &newPropobilityOfPosition);
	void setTrasitionMap(const std::vector<std::vector<int>>& newTrasitionMap);
	void setClassesParametrs(int const quantityClasses = 5,
							cv::Size const newCalsSize = cv::Size(5, 5),
							cv::Size const weigthMapSize = cv::Size(3, 3),
							const std::vector<double>& weigthsForWeigthMap = std::vector<double>{ 0.5,1.0,0.5,1.0,1.0,0.2,0.3,0.2 });
	
	cv::Mat generateImage();
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
