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
				
			}
			else
			{
				int clasNumber{ classeMap[yPointOnClasseMap][xPointOnClasseMap] };
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

void ArearsGenerate::setPropobilityOfPosition(ProbabilityOfPosition const* newPropobilityOfPosition)
{
	propobilityOfPosition = new ProbabilityOfPosition(*newPropobilityOfPosition);
	return;
}

void ArearsGenerate::setTrasitionMap(std::vector<std::vector<int>> const* newTrasitionMap)
{
	transitionMap.assign(newTrasitionMap->begin(), newTrasitionMap->end());
}

void ArearsGenerate::setSubClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	setClassesParametrs(frequencyClasses, newCalsSize, weigthMapSize, weigthsForWeigthMap);
	computeQuantityNeihbors();
	initMatVector(subClassesMasks);
}

void ArearsGenerate::setMainClassesParametrs(std::vector<int> const* frequencyClasses, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<float>* weigthsForWeigthMap)
{
	setClassesParametrs(frequencyClasses, newCalsSize, weigthMapSize, weigthsForWeigthMap);
	computeQuantityNeihbors();
	initMatVector(mainClassesMasks);
}

void ArearsGenerate::generateClasseMap()
{
	for (size_t i{ 0 }; i < classeMap[0].size(); ++i)
	{
		for (size_t j{ 0 }; j < classeMap.size(); ++j)
		{
			std::vector<double> classesProbobilityOfPosition{ propobilityOfPosition->getProbolity(j, i) };
			std::vector<float> classWeigthOnStep{ };
			classWeigthOnStep = computeFrequencyOfPosition(cv::Point(i, j));
			fromFrequencyToPropobility(&weigthsOnStep, propobilityOfNeighbors);
			for (size_t c{ 0 }; c < quantityClases; ++c)
			{
				correctionPropobilityOfNeighbors(classesProbobilityOfPosition[c], propobilityOfNeighbors[c]);
			}
			std::vector<double> propobilityOnStep{ classesProbobilityOfPosition[0] * propobilityOfNeighbors[0], classesProbobilityOfPosition[1] * propobilityOfNeighbors[1] };
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

void ArearsGenerate::initClassesMasks(std::vector<cv::Mat> &classesMasks)
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
	int color{ 20 };
	for (size_t i{ 0 }; i < quantityClases; ++i)
	{
		cv::Mat background{ mainImage.size(), CV_8UC1, cv::Scalar(color) };
		cv::bitwise_and(background, mainClassesMasks[i], background);
		cv::bitwise_or(mainImage, background, mainImage);
		color += 20;
	}
}

void ArearsGenerate::combinateMainAndSubClasses(int numberMainClass)
{

}

cv::Mat* ArearsGenerate::generateImageWithMainClasess()
{
	generateClasseMap();
	initClassesMasks(mainClassesMasks);
	initMainImage();
	return &mainImage;
}

cv::Mat* ArearsGenerate::generateImageWithSubClasess()
{
	generateClasseMap();
	initClassesMasks(subClassesMasks);
	
	initMainImage();
	return &mainImage;
}

int ArearsGenerate::getNewValue(std::vector<double>& const propobility)
{
	std::vector<int> frequencyOnStep;
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
