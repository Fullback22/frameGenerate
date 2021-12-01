#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
	enum class backgroudClases
	{
		Trees,
		Gras
	};
	std::vector<float> propabilitesInitial{0.5,0.5};
	std::vector<int> activCals{};
	std::vector<float> weigthActivCals{};
	std::vector<float> weigthBackgroudClases{};
	weigthBackgroudClases.resize(propabilitesInitial.size());
	cv::Size calsSize{};
	cv::Mat mainImage{};

}