#include "ArearsGenerate.h"

void ArearsGenerate::setClasseMapSize()
{

	int quantityCols{ mainImage.size().width / calsSize.width };
	if (mainImage.size().width % calsSize.width != 0)
	{
		++quantityCols;
	}
	int quantityRows{ mainImage.size().height / calsSize.height };
	if (mainImage.size().height % calsSize.height != 0)
	{
		++quantityRows;
	}

	classeMap.resize(quantityRows);
	for (size_t i{ 0 }; i < quantityRows; ++i)
	{
		classeMap[i].resize(quantityCols);
	}
}

void ArearsGenerate::initClasseMap()
{
	for (size_t i{ 0 }; i < classeMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < classeMap[i].size(); ++j)
		{
			classeMap[i][j] = -1;
		}
	}
}

void ArearsGenerate::computeClassesCoefficients(cv::Point const& activPoin)
{

}
