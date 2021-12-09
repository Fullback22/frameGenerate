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
	std::vector<int> frenquncesClasses{ 1,1,1,1 };
	ArearsGenerate test{ &frenquncesClasses, cv::Size(32, 40), cv::Size(500, 512) };
	test.generateClasseMap();
	test.initClassesMasks();
	test.initMainImage();
}