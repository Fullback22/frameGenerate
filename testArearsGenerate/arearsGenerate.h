#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ArearsGenerate
{
	std::vector<float> propabilitesInitial{ 0.5,0.5 };
	std::vector<std::vector<float>> weigthMap{};
	std::vector<float> weigthBackgroudClases{};
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> classesMask{};
	std::vector<std::vector<int>> classeMap{};

public:
	void setClasseMapSize();
	void initClasseMap();
	void computeClassesCoefficients(cv::Point const& activPoin);

};
