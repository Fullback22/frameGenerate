#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

void setClasseMapSize(std::vector<std::vector<int>> &inOutputMap, cv::Size const &calsSize, cv::Size const &mainImageSize)
{
	int quantityCols{ mainImageSize.width / calsSize.width };
	if (mainImageSize.width % calsSize.width != 0)
	{
		++quantityCols;
	}
	int quantityRows{ mainImageSize.height / calsSize.height };
	if (mainImageSize.height % calsSize.height !=0)
	{
		++quantityRows;
	}

	inOutputMap.resize(quantityRows);
	for (size_t i{ 0 }; i < quantityRows; ++i)
	{
		inOutputMap[i].resize(quantityCols);
	}
}

void initClasseMap(std::vector<std::vector<int>>& inOutputMap)
{
	for (size_t i{ 0 }; i < inOutputMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < inOutputMap[i].size(); ++j)
		{
			inOutputMap[i][j] = -1;
		}
	}
}

void computeClassesCoefficients(cv::Point const &activPoin, std::vector<std::vector<int>> const &inOutputMap, )
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
	std::vector<cv::Mat> classesMask{};
	std::vector<std::vector<int>> classeMap{};

}