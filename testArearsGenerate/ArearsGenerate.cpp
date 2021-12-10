#include "ArearsGenerate.h"

ArearsGenerate::ArearsGenerate(std::vector<int> const *frequencyClasses, cv::Size const newCalsSize, cv::Size const mainImageSize, cv::Size const weigthMapSize,const std::vector<float>*weigthsForWeigthMap):
	quantityClases(frequencyClasses->size()),
	quantityNotNullClasses(quantityClases),
	propabilitesOnStep(quantityClases, 0.0),
	propabilitesInitial(quantityClases, 0.0),
	calsSize(newCalsSize),
	mainImage(mainImageSize, CV_8UC1, cv::Scalar(0)),
	classesMasks(0)
{
	gen.seed(rd());
	propabilitesInitial = fromWeigthToPropabilitys(frequencyClasses);
	
	for (size_t i{0};i<quantityClases;++i)
	{
		classesMasks.push_back(cv::Mat(mainImage.size(), CV_8UC1, cv::Scalar(0)));
	}
	setClasseMapSize();
	setWeigthMapSize(weigthMapSize);
	initWeigthMap(weigthsForWeigthMap);
}

void ArearsGenerate::setMainClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size calsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	quantityMainClases = frequencyClasses->size();
	int sumWeigth{ 0 };
	for (auto& weigth : *frequencyClasses)
	{
		sumWeigth += weigth;
	}
	double averagePropability{ 1.0 / sumWeigth };
	for (size_t i{ 0 }; i < quantityMainClases; ++i)
	{
		propabilitesInitial.push_back(averagePropability * (*frequencyClasses)[i]);
		classesMasks.push_back(cv::Mat(mainImage.size(), CV_8UC1, cv::Scalar(0)));
	}
}

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

	classeMap.resize(quantityRows, std::vector<int>(quantityCols, -1));

}

void ArearsGenerate::setWeigthMapSize(cv::Size const newSize)
{
	if (newSize.height % 2 == 0)
	{
		weigthMap.resize(newSize.height + 1);
	}
	else
	{
		weigthMap.resize(newSize.height);
	}
	int quantityCols{ newSize.width };
	if (newSize.width % 2 == 0)
	{
		++quantityCols;
	}
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		weigthMap[i].resize(quantityCols);
	}
}

void ArearsGenerate::initWeigthMap(std::vector<float>const* newWeigth)
{
	size_t centerY{ weigthMap.size() / 2 };
	size_t centerX{ weigthMap[0].size() / 2 };
	size_t iterator{ 0 };
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < weigthMap[0].size(); ++j)
		{
			if (i == centerY && j == centerX)
			{
			}
			else
			{
				weigthMap[i][j] = (*newWeigth)[iterator];
				++iterator;
			}
		}
	}
}

std::vector<float> ArearsGenerate::computeClassesWeigth(cv::Point const& activPoint)
{
	std::vector<float> classesWeigth(quantityClases, 0);
	size_t yOffsetForClassesMap{ weigthMap.size() / 2 };
	size_t xOffsetForClassesMap{ weigthMap[0].size() / 2 };
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < weigthMap[0].size(); ++j)
		{
			int xPointOnClasseMap{ activPoint.x + static_cast<int>(j) - static_cast<int>(xOffsetForClassesMap) };
			int yPointOnClasseMap{ activPoint.y + static_cast<int>(i) - static_cast<int>(yOffsetForClassesMap) };
			if (xPointOnClasseMap < 0 || yPointOnClasseMap < 0 || xPointOnClasseMap >= classeMap[0].size() || yPointOnClasseMap >= classeMap.size())
			{
				for (size_t x{ 0 }; x < quantityClases; ++x)
					classesWeigth[x] += weigthMap[i][j] / quantityClases;
			}
			else
			{
				int clasNumber{ classeMap[yPointOnClasseMap][xPointOnClasseMap] };
				if (clasNumber == -1)
				{
					for (size_t x{ 0 }; x < quantityClases; ++x)
						classesWeigth[x] += weigthMap[i][j] / quantityClases;
				}
				else
				{
					classesWeigth[clasNumber] += weigthMap[i][j];
				}
			}

		}
	}

	float sumClassesWeigth{ 0.0 };
	std::for_each(classesWeigth.begin(), classesWeigth.end(), [&](float n)
		{
			sumClassesWeigth += n;
		});
	float weigthCoeficient{ 1 / sumClassesWeigth };

	for (size_t i{0};i< classesWeigth.size();++i)
	{
		classesWeigth[i] *= weigthCoeficient;
	}
	
	return classesWeigth;
}

void ArearsGenerate::computeExtensionPropabilites(std::vector<float> const* classesWeigth)
{
	quantityNotNullClasses = propabilitesInitial.size();
	float sumPropabilitiesNullClasses{ 0.0 };
	for (size_t i{0};i< classesWeigth->size();++i)
	{
		if ((*classesWeigth)[i] == 0.0)
		{
			--quantityNotNullClasses;
			sumPropabilitiesNullClasses += propabilitesInitial[i];
			propabilitesOnStep[i] = 0.0;
		}
		else
		{
			propabilitesOnStep[i] = propabilitesInitial[i];
		}
	}
	float extensionPropability{ sumPropabilitiesNullClasses / quantityNotNullClasses };
	for (auto &propability: propabilitesOnStep)
	{
		if (propability != 0.0)
			propability += extensionPropability;
	}
}

void ArearsGenerate::computeNewPropabilitys(std::vector<float> const* classesWeigth)
{
	for (size_t i{ 0 }; i < propabilitesOnStep.size(); ++i)
	{
		if (propabilitesOnStep[i] != 0)
		{
			propabilitesOnStep[i] += (*classesWeigth)[i];
			float extraPropability{ (*classesWeigth)[i] / (quantityNotNullClasses) };
			for (auto& propability : propabilitesOnStep)
			{
				if (propability != 0)
					propability -= extraPropability;
			}
		}
	}
}

void ArearsGenerate::generateClasseMap()
{

	for (size_t i{ 0 }; i < classeMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < classeMap[0].size(); ++j)
		{
			std::vector<float> classWeigthOnStep{ };
			classWeigthOnStep = computeClassesWeigth(cv::Point(j, i));
			computeExtensionPropabilites(&classWeigthOnStep);
			computeNewPropabilitys(&classWeigthOnStep);
			std::vector<int> convertedPropabilitysOnStep{};
			convertedPropabilitysOnStep = convertPropabilitysOnStepToInt();
			std::discrete_distribution<int> classDistribution{ convertedPropabilitysOnStep.begin(), convertedPropabilitysOnStep.end() };	
			classeMap[i][j] = classDistribution(gen);
		}
	}
}

std::vector<int> ArearsGenerate::convertPropabilitysOnStepToInt(int const accuracy)
{
	std::vector<int> outPropabilitys{};
	for (size_t i{ 0 }; i < propabilitesOnStep.size(); ++i)
	{
		float proabilitys{ propabilitesOnStep[i] * accuracy };
		outPropabilitys.push_back(proabilitys);
	}
	return outPropabilitys;
}

void ArearsGenerate::initClassesMasks()
{

	for (size_t i{ 0 }; i < classeMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < classeMap[0].size(); ++j)
		{
			int x1{ static_cast<int>(j) * calsSize.width };
			int x2{ (1 + static_cast<int>(j)) * calsSize.width - 1 };
			int y1{ static_cast<int>(i) * calsSize.height };
			int y2{ (1 + static_cast<int>(i)) * calsSize.height - 1 };
			if (x2 >= mainImage.size().width)
				x2 = mainImage.size().width - 1;
			if (y2 >= mainImage.size().height)
				y2 = mainImage.size().height - 1;
			cv::Point vertices[4];
			vertices[0] = cv::Point(x1, y1);
			vertices[1] = cv::Point(x1, y2);
			vertices[2] = cv::Point(x2, y2);
			vertices[3] = cv::Point(x2, y1);
			cv::fillConvexPoly(classesMasks[classeMap[i][j]], vertices, 4, cv::Scalar(255), 8);
		}
	}
}

void ArearsGenerate::initMainImage()
{
	int colors[5]{ 20, 100, 170, 200, 225 };
	for (size_t i{ 0 }; i < quantityClases; ++i)
	{
		cv::Mat background{ mainImage.size(), CV_8UC1, cv::Scalar(colors[i]) };
		cv::bitwise_and(background, classesMasks[i], background);
		cv::bitwise_or(mainImage, background, mainImage);
	}
	cv::Mat bufer{ mainImage };
	return;
}
