#include "ArearsGenerate.h"

void ArearsGenerate::computeQuantityNeihbors()
{
	quantityNeighbors = 0;
	for (auto row : weigthMap)
	{
		for (auto value : row)
		{
			if (value > 0)
				++quantityNeighbors;
		}
	}
}

void ArearsGenerate::setClassMapSize()
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

	classMap.assign(quantityRows, std::vector<int>(quantityCols, -1));
}

void ArearsGenerate::updateClassMap(const cv::Size& oldCalsSize)
{
	std::vector<std::vector<int>> buferClassMap{ classMap };
	
	int resizeCoefficientWidth{ oldCalsSize.width / calsSize.width };
	int resizeCoefficientHeigth{ oldCalsSize.height / calsSize.height };
	setClassMapSize();
	for (int i{ 0 }; i < classMap.size(); ++i)
	{
		int old_i{ i / resizeCoefficientHeigth };
		if (old_i >= buferClassMap.size())
			old_i = buferClassMap.size() - 1;
		for (int j{}; j < classMap[i].size(); ++j)
		{
			int old_j{ j / resizeCoefficientWidth };
			if (old_j >= buferClassMap[0].size())
				old_j = buferClassMap[0].size() - 1;
			classMap[i][j] = buferClassMap[old_i][old_j];
		}
	}
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

std::vector<float> ArearsGenerate::computeFrequencyOfPosition(cv::Point const& activPoint)
{
	std::vector<float> classesWeigth(quantityClasses, 0);

	weigthsOnStep.assign(quantityClasses, 0);
	size_t yOffsetForClassesMap{ weigthMap.size() / 2 };
	size_t xOffsetForClassesMap{ weigthMap[0].size() / 2 };
	activNeighbors = 0;
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < weigthMap[0].size(); ++j)
		{
			int xPointOnClasseMap{ activPoint.x + static_cast<int>(j) - static_cast<int>(xOffsetForClassesMap) };
			int yPointOnClasseMap{ activPoint.y + static_cast<int>(i) - static_cast<int>(yOffsetForClassesMap) };
			if (xPointOnClasseMap < 0 || yPointOnClasseMap < 0 || xPointOnClasseMap >= classMap[0].size() || yPointOnClasseMap >= classMap.size())
			{
				
			}
			else
			{
				int clasNumber{ classMap[yPointOnClasseMap][xPointOnClasseMap] };
				if (clasNumber == -1)
				{
					
				}
				else if(weigthMap[i][j]>0)
				{
					weigthsOnStep[clasNumber] += weigthMap[i][j];
					++activNeighbors;
				}
			}
		}
	}
	
	
	return classesWeigth;
}

void ArearsGenerate::computeExtensionWeigths(std::vector<float> const* classesWeigth)
{
	quantityNotNullClasses = weigthsInitial.size();
	float sumWeigthsNullClasses{ 0.0 };
	for (size_t i{ 0 }; i < classesWeigth->size(); ++i)
	{
		if ((*classesWeigth)[i] == 0.0)
		{
			--quantityNotNullClasses;
			sumWeigthsNullClasses += weigthsInitial[i];
			weigthsOnStep[i] = 0.0;
		}
		else
		{
			weigthsOnStep[i] = weigthsInitial[i];
		}
	}
	float extensionPropability{ sumWeigthsNullClasses / quantityNotNullClasses };
	for (auto& wigths : weigthsOnStep)
	{
		if (wigths != 0.0)
			wigths += extensionPropability;
	}
}

void ArearsGenerate::computeNewWeigths(std::vector<float> const* classesWeigth)
{
	for (size_t i{ 0 }; i < weigthsOnStep.size(); ++i)
	{
		if (weigthsOnStep[i] != 0)
		{
			weigthsOnStep[i] += (*classesWeigth)[i];
		}
	}
}

void ArearsGenerate::initMatVector(std::vector<cv::Mat>& inputVector)
{
	inputVector.clear();
	for (size_t i{ 0 }; i < quantityClasses; ++i)
	{
		inputVector.push_back(cv::Mat(mainImage.size(), CV_8UC1, cv::Scalar(0)));
	}
}

void ArearsGenerate::setClassesParametrs(int quantityClasses_, cv::Size newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	quantityClasses = quantityClasses_;
	weigthsOnStep.resize(quantityClasses, 0);
	calsSize = newCalsSize;
	setClassMapSize();
	setWeigthMapSize(weigthMapSize);
	initWeigthMap(weigthsForWeigthMap);
}



void ArearsGenerate::fromFrequencyToProbability(std::vector<int> const* frequncy, std::vector<double>& propobility)
{
	int sum{ std::accumulate(frequncy->begin(), frequncy->end(),0) };
	propobility.clear();
	for (auto &element:*frequncy)
	{
		propobility.push_back(element / (static_cast<double>(sum)+0.001));
	}
}

void ArearsGenerate::correctionProbabilityOfNeighbors(double const propobilityOfPosition, double& propobilityOfNeighbors)
{
	propobilityOfNeighbors = (static_cast<double>(quantityNeighbors - activNeighbors) / quantityNeighbors) * propobilityOfPosition + (static_cast<double>(activNeighbors) / quantityNeighbors) * propobilityOfNeighbors;
}

ArearsGenerate::ArearsGenerate(cv::Size const mainImageSize):
	mainImage(mainImageSize, CV_8UC1, cv::Scalar(0))
{
	gen.seed(rd());
}

void ArearsGenerate::setProbabilityOfPosition(ProbabilityOfPosition const* newPropobilityOfPosition)
{
	probabilityOfPosition = new ProbabilityOfPosition(*newPropobilityOfPosition);
}

void ArearsGenerate::setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap)
{
	transitionMap.assign(newTrasitionMap->begin(), newTrasitionMap->end());
}

void ArearsGenerate::setClassesParametrs(int const quantityClasses_, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	quantityClasses += quantityClasses_;
	setClassesParametrs(quantityClasses_, newCalsSize, weigthMapSize, weigthsForWeigthMap);
	computeQuantityNeihbors();
	initMatVector(classesMasks);
}

void ArearsGenerate::generateClasseMap(int const iter)
{
	generateClasseMapInitializingPart();
	generateClasseMapIterativePart(iter);
}

void ArearsGenerate::generateClasseMapInitializingPart()
{
	for (size_t i{ 0 }; i < classMap[0].size(); ++i)
	{
		for (size_t j{ 0 }; j < classMap.size(); ++j)
		{
			std::vector<double> classesProbobilityOfPosition{ probabilityOfPosition->getProbolity(j * calsSize.width, i * calsSize.height) };
			std::vector<float> classWeigthOnStep{ };
			classWeigthOnStep = computeFrequencyOfPosition(cv::Point(i, j));
			fromFrequencyToProbability(&weigthsOnStep, probabilityOfNeighbors);
			for (size_t c{ 0 }; c < quantityClasses; ++c)
			{
				correctionProbabilityOfNeighbors(classesProbobilityOfPosition[c], probabilityOfNeighbors[c]);
			}

			std::vector<double> propobilityOnStep{};
			for (size_t k{ 0 }; k < quantityClasses; ++k)
			{
				propobilityOnStep.push_back((weigthProbabilityOfNeighbors * classesProbobilityOfPosition[k] + weigthProbabilityOfPosition * probabilityOfNeighbors[k]) / (weigthProbabilityOfNeighbors + weigthProbabilityOfPosition));
			}

			classMap[j][i] = getNewValue(propobilityOnStep);
		}
	}
}

void ArearsGenerate::generateClasseMapIterativePart(int const iter)
{
	for (size_t x{ 1 }; x < iter + 1 || calsSize.width > 1; ++x)
	{
		if (x % iter == 0 && (calsSize.width > 1 || calsSize.height > 1))
		{
			cv::Size oldCalsSize{ calsSize };
			if (calsSize.width > 1)
				calsSize.width /= 2;
			if (calsSize.height > 1)
				calsSize.height /= 2;
			updateClassMap(oldCalsSize);
			x = 0;
		}

		std::vector<std::vector<int>> buferClassesMap{ classMap };
		for (size_t i{ 0 }; i < classMap[0].size(); ++i)
		{
			
			for (size_t j{ 0 }; j < classMap.size(); ++j)
			{
				std::vector<float> classWeigthOnStep{ };
				classWeigthOnStep = computeFrequencyOfPosition(cv::Point(i, j));
				
				fromFrequencyToProbability(&weigthsOnStep, probabilityOfNeighbors);

				std::vector<double> propobilityOnStep{};
				for (size_t k{ 0 }; k < quantityClasses; ++k)
				{
					propobilityOnStep.push_back(probabilityOfNeighbors[k]);
				}
				if (activNeighbors > 0 && j > 0)
				{
					for (int c{ 0 }; c < quantityClasses; ++c)
					{
						propobilityOnStep[c] *= transitionMap[classMap[j - 1][i]][c];
					}
				}
				buferClassesMap[j][i] = getNewValue(propobilityOnStep);
			}
		}
		classMap.assign(buferClassesMap.begin(), buferClassesMap.end());
	}
}

void ArearsGenerate::initClassesMasks(std::vector<cv::Mat> &classesMasks)
{
	for (size_t i{ 0 }; i < classMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < classMap[0].size(); ++j)
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
			if (classMap[i][j] > 2 || classMap[i][j] < 0)
				std::cout << "erer " << classMap[i][j] << " " << i << " " << j << std::endl;
			cv::fillConvexPoly(classesMasks[classMap[i][j]], vertices, 4, cv::Scalar(255), 8);
		}
	}
}

void ArearsGenerate::suppressionEmissions(cv::Mat& inOutputClassMap, int const kernelSize)
{
	cv::medianBlur(inOutputClassMap, inOutputClassMap, kernelSize);
}


void ArearsGenerate::initMainImage()
{
	int color{ 20 };
	for (size_t i{ 0 }; i < quantityClasses; ++i)
	{
		cv::Mat background{ mainImage.size(), CV_8UC1, cv::Scalar(color) };
		cv::bitwise_and(background, classesMasks[i], background);
		cv::bitwise_or(mainImage, background, mainImage);
		color += 20;
	}
}

cv::Mat ArearsGenerate::drawClasses(std::vector<cv::Mat>* const maskClsses)
{
	int color{ 20 };
	cv::Mat outImage{ mainImage.size(), CV_8UC1, cv::Scalar(0) };
	for (auto mask:*maskClsses)
	{
		cv::Mat background{ mainImage.size(), CV_8UC1, cv::Scalar(color) };
		cv::bitwise_and(background, mask, background);
		cv::bitwise_or(outImage, background, outImage);
		color += 20;
	}

	return outImage;
}

void ArearsGenerate::setWeigthProbabilitys(double const weigthOfPosition, double const weigthOfNeighbors)
{
	weigthProbabilityOfPosition = weigthOfPosition;
	weigthProbabilityOfNeighbors = weigthOfNeighbors;
}

cv::Mat ArearsGenerate::generateImage()
{
	generateClasseMap(20);
	initClassesMasks(classesMasks);
	cv::Mat outImage(drawClasses(&classesMasks));
	suppressionEmissions(outImage);
	return outImage;
}

int ArearsGenerate::getNewValue(std::vector<double>& const propobility)
{
	std::vector<int> frequencyOnStep;
	std::uniform_int_distribution<> initDist{ 0, static_cast<int>(quantityClasses - 1) };

	std::uniform_real_distribution<> dis{ 0.0, 1.0 };
	for (; ;)
	{
		int newValue{ initDist(gen) };
		double conversionPropability{ dis(gen) };
		if (conversionPropability < propobility[newValue])
		{
			return newValue;
		}
	}
}
