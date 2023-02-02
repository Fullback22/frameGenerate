#include "ArearsGenerate.h"

unsigned int ArearsGenerate::getTransitionCoefficient(const cv::Point& activPoint, unsigned int const targetClass)
{
	if (activPoint.y > 0)
	{
		int upperAreaClassNumber{ classMap[activPoint.y - 1][activPoint.x] };
		if (upperAreaClassNumber == -1)
		{
			if (activPoint.y < classMap.size() - 1)
			{
				int lowerAreaClassNumber{ classMap[activPoint.y + 1][activPoint.x] };
				if (lowerAreaClassNumber == -1)
					return 1;
				else
					return transitionMap[targetClass][lowerAreaClassNumber];
			}
			else
				return 1;
		}
		else
		{
			return transitionMap[upperAreaClassNumber][targetClass];
		}
	}
	else
		return 1;
}

void ArearsGenerate::computeQuantityNeihbors()
{
	quantityNeighbors = weigthMap.size() * weigthMap[0].size() - 1;
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

void ArearsGenerate::initWeigthMap(const std::vector<double>& newWeigth)
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
				weigthMap[i][j] = newWeigth[iterator];
				++iterator;
			}
		}
	}
}

void ArearsGenerate::initProbabilityOfYMap()
{
	cv::Size mapSize{ mainImage.size() };
	probabilityOfYMap = std::vector<std::vector<std::vector<double>>>(mapSize.height, std::vector<std::vector<double>>(mapSize.width, std::vector<double>(quantityClasses, 0.0)));
	for (size_t i{}; i < mapSize.height; ++i)
	{
		for (size_t j{}; j < mapSize.width; ++j)
		{
			probabilityOfYMap[i][j] = probabilityOfPosition->getProbolity(i, j);
		}
	}
	return;
}

void ArearsGenerate::computeWeigthFromPosition(const cv::Point& activPoint)
{
	weigthsOnStep.assign(quantityClasses, 0.0);
	activNeighbors = 0;
	
	size_t yOffsetForClassesMap{ weigthMap.size() / 2 };
	size_t xOffsetForClassesMap{ weigthMap[0].size() / 2 };
	
	for (size_t i{ 0 }; i < weigthMap.size(); ++i)
	{
		size_t yPointOnClasseMap{ activPoint.y + i - yOffsetForClassesMap };
		for (size_t j{ 0 }; j < weigthMap[0].size(); ++j)
		{
			size_t xPointOnClasseMap{ activPoint.x + j - xOffsetForClassesMap };
			if (xPointOnClasseMap < 0 || yPointOnClasseMap < 0 || xPointOnClasseMap >= classMap[0].size() || yPointOnClasseMap >= classMap.size() || (xPointOnClasseMap == activPoint.x && yPointOnClasseMap == activPoint.y))
			{
			}
			else
			{
				int classNumber{ classMap[yPointOnClasseMap][xPointOnClasseMap] };
				if (classNumber >= 0)
					weigthsOnStep[classNumber] += weigthMap[i][j];
				else
				{
					double oneClassWeigth = weigthMap[i][j] / weigthsOnStep.size();
					for (auto& weigth : weigthsOnStep)
					{
						weigth += oneClassWeigth;
					}
				}
				++activNeighbors;
			}
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

double ArearsGenerate::correctionCoefficientOfNeighbors(double const coefficientOfPosition, double const coefficientOfNeighbors)
{
	return  (static_cast<double>(quantityNeighbors - activNeighbors) / quantityNeighbors) * coefficientOfPosition + (static_cast<double>(activNeighbors) / quantityNeighbors) * coefficientOfNeighbors;
}

double ArearsGenerate::correctionCoefficientOf_Y(double const coefficientOfPosition, double const coefficientOfNeighbors)
{
	return  (static_cast<double>(quantityNeighbors - activNeighbors) / quantityNeighbors) * coefficientOfPosition + (static_cast<double>(activNeighbors) / quantityNeighbors) * coefficientOfNeighbors;
}

ArearsGenerate::ArearsGenerate(cv::Size const mainImageSize):
	mainImage(mainImageSize, CV_8UC1, cv::Scalar(0)),
	numberGenerator(randomDevice())
{
}

void ArearsGenerate::setProbabilityOfPosition(const ProbabilityOfPosition& newPropobilityOfPosition)
{
	probabilityOfPosition = new ProbabilityOfPosition(newPropobilityOfPosition);
}

void ArearsGenerate::setTrasitionMap(const std::vector<std::vector<int>>& newTrasitionMap)
{
	transitionMap.assign(newTrasitionMap.begin(), newTrasitionMap.end());
}

void ArearsGenerate::setClassesParametrs(int const quantityClasses_, cv::Size const newCalsSize, cv::Size const weigthMapSize, const std::vector<double>& weigthsForWeigthMap)
{
	quantityClasses = quantityClasses_;
	weigthsOnStep.resize(quantityClasses, 0);
	calsSize = newCalsSize;
	setClassMapSize();
	setWeigthMapSize(weigthMapSize);
	initWeigthMap(weigthsForWeigthMap);


	computeQuantityNeihbors();
}

void ArearsGenerate::generateClasseMap(size_t const iter)
{
	//bool isFirstStep{ true };
	initProbabilityOfYMap();
	for (size_t z{ 1 }; z < iter + 1 || calsSize.width > 1; ++z)
	{
		if (z % iter == 0 && (calsSize.width > 1 || calsSize.height > 1))
		{
			cv::Size oldCalsSize{ calsSize };
			if (calsSize.width > 1)
				calsSize.width /= 2;
			if (calsSize.height > 1)
				calsSize.height /= 2;
			updateClassMap(oldCalsSize);
			z = 0;
		}

		//std::vector<std::vector<int>> newClassesMap{ classMap };

		cv::Point activPoint{};
		activPoint.x = static_cast<int>(ceil(classMap[0].size() / 2.0));
		if (classMap[0].size() % 2 != 0)
			--activPoint.x;
		activPoint.y = static_cast<int>(ceil(classMap.size() / 2.0));
		if (classMap.size() % 2 != 0)
			--activPoint.y;
		cv::Rect imageRect{ cv::Point{0, 0 }, cv::Size(classMap[0].size(), classMap.size()) };

		bool activCoordinateIs_Y{ false };
		int iterator{ -1 };
		int step{ 0 };
		int i{ -1 };
		/*if (z % 2 == 0)
		{
			i = classMap[0].size()-1;
		}
		else
			iterator = 1;

		for (; i < classMap[0].size() && i>=0; i+=iterator)
		{
			for (size_t j{ 0 }; j < classMap.size(); ++j)
			{
				computeNewClassInPosition(cv::Point{ static_cast<int>(i), static_cast<int>(j) });
			}
		}*/

		cv::Mat test{ imageRect.size(), CV_8UC1 };
		char color{};
		for (int numberUpdatePixels{ imageRect.area() }; numberUpdatePixels > 0; )
		{
			bool isPointContainsInRect{};
			for (; i < step; ++i)
			{
				if (activCoordinateIs_Y)
					activPoint.y += iterator;
				else
					activPoint.x += iterator;

				isPointContainsInRect = imageRect.contains(activPoint);
				if (isPointContainsInRect)
				{
					test.at<uchar>(activPoint) = color;
					++color;
					//if(isFirstStep)
						computeNewClassInPosition(activPoint);
					/*else
						computeNewClassInPosition(activPoint, &newClassesMap);*/
					--numberUpdatePixels;
				}
				else
				{
					i = step;
					if (activCoordinateIs_Y)
						activPoint.x += step * iterator * -1;
					else
						activPoint.y += (step + 1) * iterator;
				}
			}

			if (!isPointContainsInRect)
			{
				++step;
				iterator *= -1;
			}
			else
			{
				if (activCoordinateIs_Y)
					iterator *= -1;
				else
					++step;
				activCoordinateIs_Y = !activCoordinateIs_Y;
			}
			i = 0;
		}
		/*if (!isFirstStep)
			classMap.assign(newClassesMap.begin(), newClassesMap.end());
		else
			isFirstStep = false;*/

		std::vector<cv::Mat> classesMasks;
		initMatVector(classesMasks);
		initClassesMasks(classesMasks);

		cv::Mat outImage{ drawClasses(&classesMasks) };
		cv::imshow("qwewr", outImage);
		cv::waitKey(23);
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
			
			cv::fillConvexPoly(classesMasks[classMap[i][j]], vertices, 4, cv::Scalar(255), 8);
		}
	}
	cv::Mat test{ classesMasks[0] };
	return ;
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

cv::Mat ArearsGenerate::generateImage()
{
	generateClasseMap(10);
	std::vector<cv::Mat> classesMasks;
	initMatVector(classesMasks);
	initClassesMasks(classesMasks);
	
	cv::Mat outImage{ drawClasses(&classesMasks) };
	return outImage;
}

int ArearsGenerate::getNewValue(std::vector<double>& const propobility)
{
	std::uniform_int_distribution<> initDist{ 0, static_cast<int>(quantityClasses - 1) };

	std::uniform_real_distribution<> dis{ 0.0, 1.0 };
	for (; ;)
	{
		int newValue{ initDist(numberGenerator) };
		double conversionPropability{ dis(numberGenerator) };
		if (conversionPropability < propobility[newValue])
		{
			return newValue;
		}
	}
}

void ArearsGenerate::computeNewClassInPosition(const cv::Point& position, std::vector<std::vector<int>>* updatedClassMap)
{
	computeWeigthFromPosition(position);
	std::vector<double> classesCoefficientOfNeighbors{ fromWeigthToProbabilitys(weigthsOnStep) };
	std::vector<double> classesCoefficientFrom_Y{ probabilityOfYMap[position.y * calsSize.height][position.x * calsSize.width] };
	std::vector<int> classesCoefficientTransition(quantityClasses, 0);
	for (size_t c{ 0 }; c < quantityClasses; ++c)
	{
		classesCoefficientTransition[c] = getTransitionCoefficient(position, c);
		classesCoefficientOfNeighbors[c] = correctionCoefficientOfNeighbors(classesCoefficientFrom_Y[c], classesCoefficientOfNeighbors[c]);
		classesCoefficientFrom_Y[c] = correctionCoefficientOf_Y(classesCoefficientFrom_Y[c], classesCoefficientOfNeighbors[c]);
	}

	std::vector<double> classesWeigth(quantityClasses, 0.0);

	for (size_t k{ 0 }; k < quantityClasses; ++k)
	{
		classesWeigth[k] = classesCoefficientOfNeighbors[k] * classesCoefficientFrom_Y[k] * classesCoefficientTransition[k];
	}

	std::vector<double> propobilityOnStep{ fromWeigthToProbabilitys(classesWeigth) };
	if(updatedClassMap == nullptr)
		classMap[position.y][position.x] = getNewValue(propobilityOnStep);
	else
		(*updatedClassMap)[position.y][position.x] = getNewValue(propobilityOnStep);
}
