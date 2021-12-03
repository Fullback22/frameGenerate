#include "arearsGenerate.h"



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


int main()
{
	enum class backgroudClases
	{
		Trees,
		Gras
	};
	std::vector<float> propabilitesInitial{ 0.5,0.5 };
	std::vector<int> activCals{};
	std::vector<std::vector<int>> classeMap_(5, std::vector<int>(5, 5));
	std::vector<float> weigthActivCals{};
	std::vector<float> weigthBackgroudClases{};
	weigthBackgroudClases.resize(propabilitesInitial.size(), -1);
	cv::Size calsSize{};
	cv::Mat mainImage{};
	std::vector<cv::Mat> classesMask{};
	std::vector<std::vector<int>> classeMap{};


	std::vector<int> frenquncesClasses{ 1,1,1,1 };
	ArearsGenerate test{ &frenquncesClasses, cv::Size(8, 8), cv::Size(512, 512) };
	test.generateClasseMap();
}