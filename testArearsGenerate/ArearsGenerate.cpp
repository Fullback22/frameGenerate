#include "ArearsGenerate.h"

void ArearsGenerate::computeQuantityNeihbors()
{
	for (auto row : weigthMap)
	{
		for (auto value : row)
		{
			if (value > 0)
				++quantityNeighbors;
		}
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

std::vector<float> ArearsGenerate::computeFrequencyOfPosition(cv::Point const& activPoint)
{
	std::vector<float> classesWeigth(quantityClases, 0);

	weigthsOnStep.assign(weigthsInitial.begin(), weigthsInitial.end());

	size_t yOffsetForClassesMap{ weigthMap.size() / 2 };
	size_t xOffsetForClassesMap{ weigthMap[0].size() / 2 };
	activNeighbors = 0;
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		for (size_t j{ 0 }; j < weigthMap[0].size(); ++j)
		{
			int xPointOnClasseMap{ activPoint.x + static_cast<int>(j) - static_cast<int>(xOffsetForClassesMap) };
			int yPointOnClasseMap{ activPoint.y + static_cast<int>(i) - static_cast<int>(yOffsetForClassesMap) };
			if (xPointOnClasseMap < 0 || yPointOnClasseMap < 0 || xPointOnClasseMap >= classeMap[0].size() || yPointOnClasseMap >= classeMap.size())
			{
				/*for (size_t x{ 0 }; x < quantityClases; ++x)
					classesWeigth[x] += weigthMap[i][j] / quantityClases;*/
			}
			else
			{
				int clasNumber{ classeMap[yPointOnClasseMap][xPointOnClasseMap] };
				if (clasNumber == -1)
				{
					/*for (size_t x{ 0 }; x < quantityClases; ++x)
						classesWeigth[x] += weigthMap[i][j] / quantityClases;*/
				}
				else if(weigthMap[i][j]>0)
				{
					weigthsOnStep[clasNumber] += weigthMap[i][j];

					++activNeighbors;
					//classesWeigth[clasNumber] += weigthMap[i][j];
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

std::vector<int> ArearsGenerate::convertPropabilitysOnStepToInt(int const accuracy)
{
	std::vector<int> outPropabilitys{};
	for (size_t i{ 0 }; i < weigthsOnStep.size(); ++i)
	{
		/*float proabilitys{ weigthsOnStep[i] * accuracy };
		outPropabilitys.push_back(proabilitys);*/
	}
	return outPropabilitys;
}

void ArearsGenerate::initMatVector(std::vector<cv::Mat>& inputVector)
{
	inputVector.clear();
	for (size_t i{ 0 }; i < quantityClases; ++i)
	{
		inputVector.push_back(cv::Mat(mainImage.size(), CV_8UC1, cv::Scalar(0)));
	}
}

void ArearsGenerate::setClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	quantityClases = frequencyClasses->size();

	weigthsOnStep.resize(quantityClases, 0);
	weigthsInitial.assign(frequencyClasses->begin(), frequencyClasses->end());
	fromFrequencyToPropobility(frequencyClasses, startPropobility);
	calsSize = newCalsSize;

	setClasseMapSize();
	setWeigthMapSize(weigthMapSize);
	initWeigthMap(weigthsForWeigthMap);
}



void ArearsGenerate::fromFrequencyToPropobility(std::vector<int> const* frequncy, std::vector<double>& propobility)
{
	int sum{ std::accumulate(frequncy->begin(), frequncy->end(),0) };
	propobility.clear();
	for (auto &element:*frequncy)
	{
		propobility.push_back(element / static_cast<double>(sum));
	}
}

void ArearsGenerate::fromPropobilityToFrequency(std::vector<double> const* propobility, std::vector<int>& frequncy, int accurusy)
{
	frequncy.clear();
	for (auto& element : *propobility)
	{
		frequncy.push_back(element * accurusy);
	}
}

void ArearsGenerate::correctionPropobilityOfNeighbors(double const propobilityOfPosition, double& propobilityOfNeighbors)
{
	propobilityOfNeighbors = (static_cast<double>(quantityNeighbors - activNeighbors) / quantityNeighbors) * propobilityOfPosition + (static_cast<double>(activNeighbors) / quantityNeighbors) * propobilityOfNeighbors;
}

ArearsGenerate::ArearsGenerate(cv::Size const mainImageSize):
	mainImage(mainImageSize, CV_8UC1, cv::Scalar(0))
{
	gen.seed(rd());
}

void ArearsGenerate::setPropobilityOfPosition(std::vector<std::vector<double>> const* newPropobilityOfPosition)
{
	propobilityOfPosition.assign(newPropobilityOfPosition->begin(), newPropobilityOfPosition->end());
}

void ArearsGenerate::setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap)
{
	transitionMap.assign(newTrasitionMap->begin(), newTrasitionMap->end());
}

void ArearsGenerate::setSubClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	setClassesParametrs(frequencyClasses, newCalsSize, weigthMapSize, weigthsForWeigthMap);
	initMatVector(subClassesMasks);
}

void ArearsGenerate::setMainClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	setClassesParametrs(frequencyClasses, newCalsSize, weigthMapSize, weigthsForWeigthMap);
	computeQuantityNeihbors();
	initMatVector(mainClassesMasks);
}

void ArearsGenerate::generateMainClasseMap()
{
	for (size_t i{ 0 }; i < classeMap[0].size(); ++i)
	{
		for (size_t j{ 0 }; j < classeMap.size(); ++j)
		{
			std::vector<float> classWeigthOnStep{ };
			classWeigthOnStep = computeFrequencyOfPosition(cv::Point(i, j));
			fromFrequencyToPropobility(&weigthsOnStep, propobilityOfNeighbors);
			for (size_t c{ 0 }; c < quantityClases; ++c)
			{
				correctionPropobilityOfNeighbors(propobilityOfPosition[c][j * calsSize.height], propobilityOfNeighbors[c]);
			}
			std::vector<double> propobilityOnStep{ propobilityOfPosition[0][j * calsSize.height] * propobilityOfNeighbors[0], propobilityOfPosition[1][j * calsSize.height] * propobilityOfNeighbors[1] };
			if (activNeighbors > 0 && j > 0)
			{
				for (int c{ 0 }; c < quantityClases; ++c)
				{
					propobilityOnStep[c] *= transitionMap[classeMap[j - 1][i]][c];
				}
			}
			classeMap[j][i] = getNewValue(propobilityOnStep);		
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
			classWeigthOnStep = computeFrequencyOfPosition(cv::Point(j, i));
			computeExtensionWeigths(&classWeigthOnStep);
			computeNewWeigths(&classWeigthOnStep);
			std::vector<int> convertedPropabilitysOnStep{};
			convertedPropabilitysOnStep = convertPropabilitysOnStepToInt();
			std::discrete_distribution<int> classDistribution{ convertedPropabilitysOnStep.begin(), convertedPropabilitysOnStep.end() };	
			classeMap[i][j] = classDistribution(gen);
		}
	}
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
			cv::fillConvexPoly(mainClassesMasks[classeMap[i][j]], vertices, 4, cv::Scalar(255), 8);
		}
	}
}

void ArearsGenerate::initMainImage()
{
	int colors[5]{ 20, 100, 170, 200, 225 };
	for (size_t i{ 0 }; i < quantityClases; ++i)
	{
		cv::Mat background{ mainImage.size(), CV_8UC1, cv::Scalar(colors[i]) };
		cv::bitwise_and(background, mainClassesMasks[i], background);
		cv::bitwise_or(mainImage, background, mainImage);
	}
	cv::Mat bufer{ mainImage };
	//cv::imwrite("myModel.png", mainImage);
	return;
}

int ArearsGenerate::getNewValue(std::vector<double>& const propobility)
{
	std::vector<int> frequencyOnStep;
	//fromPropobilityToFrequency(&propobility, frequencyOnStep);
	//std::discrete_distribution<int> classDistribution{ frequencyOnStep.begin(), frequencyOnStep.end() };
	std::uniform_int_distribution<> initDist{ 0, static_cast<int>(quantityClases - 1) };

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
