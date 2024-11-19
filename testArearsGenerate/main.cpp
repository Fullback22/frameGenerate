#include "ArearsGenerate.h"
#include "AreaParametr.h"
#include "TextureParametr.h"
#include "TextureSynthesis.h"
#include <Windows.h>
#include <vector>
#include <list>
#include <map>
#include <algorithm>

void getBoundingBoxAndSetMask(cv::Rect2i& boundingBox, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint);
void toWest(int& minX, int const replaceableColor, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points);
void toEast(int& maxX, int const replaceableColor, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points);

void getQuantityClasses(const cv::Mat& arearsMap, std::vector<int>& classes);

void readTextureClasses(const std::string name, std::map<std::string, int>& classes, std::map<std::string, cv::Mat>& textureMask);
void repaintMap(cv::Mat& arearsMap, std::map<std::string, int>& classes, const int startColor = 0);

void setTexture(cv::Mat& outputImage, const cv::Mat& textureImage, cv::Mat& mask, const cv::Rect2i& boundingBox);
std::string getClassName(const std::map<std::string, int>& classes, const int value);


int main()
{
	AreaParametr param{ "modelParametrs.json" };
	TextureParametr paramTexture{ "modelParametrs.json" };


	std::vector<cv::Size> standartImageSize{ {640, 480}, {800,600}, {960, 540}, {1024, 600}, {1280, 720}, {1280, 1024}, {1600, 900}, {1920, 1080}, {2048,1080} };
	int quantityOfSize{ static_cast<int>(standartImageSize.size()) };

	std::uniform_int_distribution<> imageSizeDistr{ 0, quantityOfSize - 1 };
	std::random_device rd{};
	std::mt19937 generator{ rd() };

    for (int i{ 0 }; i < param.quantityImage; ++i)
    {
        const int numberOfImageSize{ imageSizeDistr(generator) };
        const cv::Size& imageSize{ standartImageSize[numberOfImageSize] };

        int landAirBorder{ static_cast<int>(imageSize.height * param.landAirProportion) };
        int landAirSigma = landAirBorder * 0.1;

        std::uniform_int_distribution<> initDist{ landAirBorder - landAirSigma, landAirBorder + landAirSigma };
        double positionOffset{ static_cast<double>(initDist(generator)) };


        std::vector<std::vector<double>> probabilityOfPosition(param.quantityClasses, std::vector<double>(imageSize.height));

        param.setProbabilityOfPosition(probabilityOfPosition, positionOffset);

        ProbabilityOfPosition probobility{ param.lowerOffsetValue, param.upperOffsetValue, imageSize.width / param.upperOffsetUpdateValue, imageSize.width / param.lowerOffsetUpdateValue, 0.2, param.multiplicityResetToZeroOffset };
        probobility.setProbability(probabilityOfPosition);

        ArearsGenerate myModel{ imageSize };
        myModel.setClassesParametrs(param.quantityClasses, param.callSize, param.weigthMapSize, param.weigthsForWeigthMap);
        myModel.setProbabilityOfPosition(probobility);
        myModel.setTrasitionMap(param.transitionMap);

        cv::Mat imageWithMainClasses(myModel.generateImage());

        std::string mainImageName{ "myModel_areas/myModel_" + std::to_string(i + param.startNumber) + ".png" };
        cv::imwrite(mainImageName, imageWithMainClasses);
        std::cout << i + param.startNumber << std::endl;



        cv::Mat inputImage{ cv::imread(mainImageName, 0) };

        
        std::map<std::string, cv::Mat> textureImage;
        
        repaintMap(inputImage, paramTexture.classes);


        cv::Mat outputImage{ inputImage.size(), CV_8UC1, cv::Scalar{ 255 } };
        TextureSynthesis ts{};
        cv::Size2i testureBlock{ 50, 50 };
        ts.setBlockSize(testureBlock);

        for (auto& n : paramTexture.classes)
        {
            readTextureClasses(n.first, paramTexture.classes, textureImage);
        }

        unsigned int useColor{ 255 };
        unsigned int useColor2{ 200 };
        for (size_t i{}; i < inputImage.rows; ++i)
        {
            for (size_t j{}; j < inputImage.cols; ++j)
            {
                if (inputImage.at<uchar>(i, j) != useColor && inputImage.at<uchar>(i, j) != useColor2)
                {
                    cv::Rect2i boundingBox{};
                    std::string className{ getClassName(paramTexture.classes, inputImage.at<uchar>(i, j)) };

                    getBoundingBoxAndSetMask(boundingBox, useColor, inputImage, cv::Point2i(j, i));
                    cv::Mat texture{ boundingBox.size(), CV_8UC1, cv::Scalar{double(paramTexture.classes[className])} };
                    if (textureImage.find(className) != textureImage.end())
                    {
                        cv::Mat baseImage{ cv::imread(className + ".png") };
                        ts.setOutputSize(boundingBox.size());
                        ts.setBaseImage(baseImage, textureImage[className]);
                        ts.generateTexture();
                        ts.getMaskImage(texture);
                    }
                    setTexture(outputImage, texture, inputImage, boundingBox);
                    cv::waitKey();
                }
            }
        }
        cv::Mat test(cv::imread("myModel_13.png", 0));
        cv::imwrite("outImage.png", outputImage);

        std::ofstream fileWithClasse{ "outImage.txt" };
        if (fileWithClasse.is_open())
        {
            for (auto& n : paramTexture.classes)
            {
                fileWithClasse << n.second << '\t';
                fileWithClasse << n.first << '\n';
            }
            fileWithClasse.close();
        }
    }
	return 0;
}


void getBoundingBoxAndSetMask(cv::Rect2i& boundingBox, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint)
{
    int minX{ startPoint.x };
    int minY{ startPoint.y };
    int maxX{ startPoint.x };
    int maxY{ startPoint.y };

    unsigned int replaceableColor{ image.at<uchar>(startPoint) };
    std::list<cv::Point2i> points;
    points.push_front(startPoint);

    cv::Point2i activPoint{ };
    while (points.size() > 0)
    {
        activPoint = points.back();
        if (image.at<uchar>(activPoint) == replaceableColor)
        {
            if (minY > activPoint.y)
            {
                minY = activPoint.y;
            }
            if (maxY < activPoint.y)
            {
                maxY = activPoint.y;
            }
            toWest(minX, replaceableColor, maskColor, image, activPoint, points);
            toEast(maxX, replaceableColor, maskColor, image, activPoint, points);
        }
        points.pop_back();
    }
    boundingBox.x = minX;
    boundingBox.y = minY;
    boundingBox.width = maxX - minX + 1;
    boundingBox.height = maxY - minY + 1;
}

void toWest(int& minX, int const replaceableColor, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points)
{
    cv::Point2i west{ startPoint };

    for (; west.x >= 0; --west.x)
    {
        if (image.at<uchar>(west) == replaceableColor)
        {
            if (minX > west.x)
            {
                minX = west.x;
            }
            image.at<uchar>(west) = maskColor;
            cv::Point2i north{ west };
            --north.y;
            if (north.y >= 0)
            {
                if (image.at<uchar>(north) == replaceableColor)
                {
                    points.push_front(north);
                }
            }
            cv::Point2i south{ west };
            ++south.y;
            if (south.y < image.rows)
            {
                if (image.at<uchar>(south) == replaceableColor)
                {
                    points.push_front(south);
                }
            }
        }
        else
        {
            return;
        }
    }
}

void toEast(int& maxX, int const replaceableColor, const int maskColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points)
{
    cv::Point2i east{ startPoint };
    ++east.x;
    for (; east.x < image.cols; ++east.x)
    {
        if (image.at<uchar>(east) == replaceableColor)
        {
            if (maxX < east.x)
            {
                maxX = east.x;
            }
            image.at<uchar>(east) = maskColor;
            cv::Point2i north{ east };
            --north.y;
            if (north.y >= 0)
            {
                if (image.at<uchar>(north) == replaceableColor)
                {
                    points.push_front(north);
                }
            }
            cv::Point2i south{ east };
            ++south.y;
            if (south.y < image.rows)
            {
                if (image.at<uchar>(south) == replaceableColor)
                {
                    points.push_front(south);
                }
            }
        }
        else
        {
            return;
        }
    }
}

void getQuantityClasses(const cv::Mat& arearsMap, std::vector<int>& classes)
{
    unsigned int previeClass{ 260 };
    for (size_t i{}; i < arearsMap.rows; ++i)
    {
        for (size_t j{}; j < arearsMap.cols; ++j)
        {
            if (previeClass != arearsMap.at<uchar>(i, j))
            {
                if (std::find(classes.begin(), classes.end(), arearsMap.at<uchar>(i, j)) == classes.end())
                {
                    previeClass = arearsMap.at<uchar>(i, j);
                    classes.push_back(arearsMap.at<uchar>(i, j));
                }
            }
        }
    }
}

void readTextureClasses(const std::string name, std::map<std::string, int>& classes, std::map<std::string, cv::Mat>& textureMask)
{
    std::string format{ ".txt" };
    std::ifstream fileWithClasse{ name + format };
    std::map<std::string, int> textureClasses;
    if (fileWithClasse.is_open())
    {
        while (true)
        {
            int classColor;
            std::string className;
            fileWithClasse >> classColor;
            fileWithClasse >> className;
            if (fileWithClasse.eof())
            {
                break;
            }
            else
            {
                textureClasses[className] = classColor;
            }
        }
        fileWithClasse.close();
        std::string maskFormat{ "Mask.png" };
        textureMask[name] = cv::imread(name + maskFormat, 0);
        repaintMap(textureMask[name], textureClasses, classes.size());
    }
    for (auto& n : textureClasses)
        classes.insert(n);
}

void repaintMap(cv::Mat& arearsMap, std::map<std::string, int>& classes, const int startColor)
{
    std::vector<std::pair<int, int>> cls(classes.size());
    int color{ startColor };
    int i{  };
    for (auto& n : classes)
    {
        cls[i].first = n.second;
        cls[i].second = color;
        n.second = color;
        ++i;
        ++color;
    }

    const std::pair<int, int>* activPair{ &cls[0] };
    for (size_t i{}; i < arearsMap.rows; ++i)
    {
        for (size_t j{}; j < arearsMap.cols; ++j)
        {
            if (activPair->first != arearsMap.at<uchar>(i, j))
            {
                for (auto& p : cls)
                {
                    if (p.first == arearsMap.at<uchar>(i, j))
                    {
                        activPair = &p;
                        break;
                    }
                }
            }
            arearsMap.at<uchar>(i, j) = activPair->second;
        }
    }
}

void setTexture(cv::Mat& outputImage, const cv::Mat& textureImage, cv::Mat& mask, const cv::Rect2i& boundingBox)
{
    for (size_t i{}; i < boundingBox.height; ++i)
    {
        for (size_t j{}; j < boundingBox.width; ++j)
        {
            if (outputImage.at<uchar>(i + boundingBox.y, j + boundingBox.x) == 255 && mask.at<uchar>(i + boundingBox.y, j + boundingBox.x) == 255)
            {
                outputImage.at<uchar>(i + boundingBox.y, j + boundingBox.x) = textureImage.at<uchar>(i, j);
                mask.at<uchar>(i + boundingBox.y, j + boundingBox.x) = 200;
            }
        }
    }
}

std::string getClassName(const std::map<std::string, int>& classes, const int value)
{
    for (const auto& n : classes)
    {
        if (n.second == value)
            return n.first;
    }
}
