#include <random>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ArearsGenerate
{
	size_t quantityClases{};
	size_t quantityNotNullClasses{};
	std::vector<float> propabilitesOnStep{};
	std::vector<float> propabilitesInitial{};
	
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> classesMasks{};
	
	std::vector<std::vector<float>> weigthMap{};
	std::vector<std::vector<int>> classeMap{};

	std::random_device rd;
	std::mt19937 gen;

	void setClasseMapSize();
	//void initClasseMap();
	void setWeigthMapSize(cv::Size const newSize = cv::Size(3, 3));
	void initWeigthMap(std::vector<float>const *newWeigth);
	std::vector<float> computeClassesWeigth(cv::Point const& activPoint);
	void computeExtensionPropabilites(std::vector<float> const* classesWeigth);
	void computeNewPropabilitys(std::vector<float> const* classesWeigth);
	std::vector<int> convertPropabilitysOnStepToInt(int const accuracy = 1000);
	
public:
	ArearsGenerate(std::vector<int> const* frequencyClasses,
					cv::Size const calsSize,
					cv::Size const mainImageSize,
					cv::Size const weigthMapSize = cv::Size(3, 3),
					const std::vector<float>* weigthsForWeigthMap = new std::vector<float>{1,1,1,0,0,0,0,0});
	
	void generateClasseMap();
	void initClassesMasks();
	void initMainImage();
};
