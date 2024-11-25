#include "SettingsTexture.h"

SettingsTexture::SettingsTexture(const std::string& fileName)
{
    if (fileName != "")
    {
        json modelParametrs;

        std::ifstream paramFile(fileName);

        if (!paramFile.is_open()) {
            std::cout << "ERROR: File not opened" << std::endl;
        }
        else
        {
            std::cout << "OK: File opened" << std::endl;

            try
            {
                modelParametrs = json::parse(paramFile);
            }
            catch (...)
            {
                std::cout << "ERROR: JSON OPEN" << std::endl;
                paramFile.close();
            }
            try
            {
                json channelJson = modelParametrs.at("TEXTURE_PARAMETRS");
                setMainClasses(channelJson);
                textureBlock.width = channelJson.at("textureBlockWidth").get<int>();
                textureBlock.height = channelJson.at("textureBlockHeigth").get<int>();
            }
            catch (...)
            {
                std::cout << "ERROR: JSON READ" << std::endl;
                paramFile.close();
            }
            paramFile.close();
        }
        quantityMainClasses = classes.size();
        setClassesTexture();
    }
    return;
}

void SettingsTexture::repaintImage(cv::Mat& arearsMap, const std::map<std::string, int>& targetClasses, const std::map<std::string, int>& originalClasses)
{
    std::vector<std::pair<int, int>> transitionMap;

    for (const auto& n : originalClasses)
    {
        transitionMap.push_back(std::pair<int, int>(n.second, targetClasses.at(n.first)));
    }

    const std::pair<int, int>* activPair{ &transitionMap[0] };
    for (size_t i{}; i < arearsMap.rows; ++i)
    {
        for (size_t j{}; j < arearsMap.cols; ++j)
        {
            if (activPair->first != arearsMap.at<uchar>(i, j))
            {
                for (const auto& p : transitionMap)
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

std::string SettingsTexture::getRandomTexture(const std::string& textureName)
{
    std::string dirName{ textureDirectory + "/" + textureName };
    int fileCount{};
    if (fs::is_directory(dirName))
    {
        for (const auto& entry : fs::directory_iterator(dirName))
        {
            if (entry.is_regular_file())
            {
                ++fileCount;
            }
        }


        if (fileCount % 2 != 1)
        {
            std::cout << "Incorect file struct in " << textureName << "texture\n";
        }
        int quantityTextureImage{ fileCount / 2 };
        std::uniform_int_distribution<> imageSizeDistr{ 0, quantityTextureImage - 1 };
        std::random_device rd{};
        std::mt19937 generator{ rd() };
        int textureNumber{ imageSizeDistr(generator) };
        return textureName + "_" + std::to_string(textureNumber);
    }
    return std::string{};
}

void SettingsTexture::setMainClasses(const json& channelJson)
{
    std::vector<std::string> buferForClassNames;
    buferForClassNames = channelJson.at("mainClasses").get<std::vector<std::string>>();
    int color{ 0 };
    int const colorStep{ 1 };
    for (const auto& className : buferForClassNames)
    {
        classes[className] = color;
        color += colorStep;
    }
}

void SettingsTexture::setClassesTexture()
{
    for (const auto& mainClass : classes)
    {
        std::string mapLegendName{ textureDirectory + "/" + mainClass.first + "/" + mainClass.first + ".txt"};
        std::ifstream fileWithClasse{ mapLegendName };

        if (fileWithClasse.is_open())
        {
            while (true)
            {
                int classColor{};
                std::string className{};
                fileWithClasse >> classColor;
                fileWithClasse >> className;
                if (fileWithClasse.eof())
                {
                    break;
                }
                else
                {
                    subClasses[className] = classColor;
                    classes[className] = classes.size();
                }
            }
            fileWithClasse.close();
        }
    }
}

std::string SettingsTexture::getClassName(const int value) const
{
    for (const auto& n : classes)
    {
        if (n.second == value)
            return n.first;
    }
}

void SettingsTexture::getBoundingBoxAndSetMask(cv::Rect2i& boundingBox, const cv::Point2i& startPoint)
{
    int minX{ startPoint.x };
    int minY{ startPoint.y };
    int maxX{ startPoint.x };
    int maxY{ startPoint.y };

    unsigned int replaceableColor{ mapImage.at<uchar>(startPoint) };
    std::list<cv::Point2i> points;
    points.push_front(startPoint);

    cv::Point2i activPoint{ };
    while (points.size() > 0)
    {
        activPoint = points.back();
        if (mapImage.at<uchar>(activPoint) == replaceableColor)
        {
            if (minY > activPoint.y)
            {
                minY = activPoint.y;
            }
            if (maxY < activPoint.y)
            {
                maxY = activPoint.y;
            }
            toWest(minX, replaceableColor, mapImage, activPoint, points);
            toEast(maxX, replaceableColor, mapImage, activPoint, points);
        }
        points.pop_back();
    }
    boundingBox.x = minX;
    boundingBox.y = minY;
    boundingBox.width = maxX - minX + 1;
    boundingBox.height = maxY - minY + 1;
}

void SettingsTexture::toWest(int& minX, int const replaceableColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points)
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

void SettingsTexture::toEast(int& maxX, int const replaceableColor, cv::Mat& image, const cv::Point2i& startPoint, std::list<cv::Point2i>& points)
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

void SettingsTexture::setTexture(const cv::Mat& textureImage, const cv::Rect2i& boundingBox)
{
    for (size_t i{}; i < boundingBox.height; ++i)
    {
        for (size_t j{}; j < boundingBox.width; ++j)
        {
            if (mapImageWithTexture.at<uchar>(i + boundingBox.y, j + boundingBox.x) == 255 && mapImage.at<uchar>(i + boundingBox.y, j + boundingBox.x) == maskColor)
            {
                mapImageWithTexture.at<uchar>(i + boundingBox.y, j + boundingBox.x) = textureImage.at<uchar>(i, j);
                mapImage.at<uchar>(i + boundingBox.y, j + boundingBox.x) = 200;
            }
        }
    }
}

void SettingsTexture::setMapImage(const cv::Mat& image)
{
    image.copyTo(mapImage);
}

void SettingsTexture::addTextureToMapImage()
{
    mapImageWithTexture = cv::Mat(mapImage.size(), CV_8UC1, cv::Scalar{ 255 });
    TextureSynthesis ts{};
    ts.setBlockSize(textureBlock);

    unsigned int useColor2{ 200 };
    for (size_t i{}; i < mapImage.rows; ++i)
    {
        for (size_t j{}; j < mapImage.cols; ++j)
        {
            if (mapImage.at<uchar>(i, j) != maskColor && mapImage.at<uchar>(i, j) != useColor2)
            {
                cv::Rect2i boundingBox{};
                std::string className{ getClassName(mapImage.at<uchar>(i, j)) };

                getBoundingBoxAndSetMask(boundingBox, cv::Point2i(j, i));
                cv::Mat texture{ boundingBox.size(), CV_8UC1, cv::Scalar{double(classes[className])} };
                if (textureImage.find(className) != textureImage.end())
                {
                    ts.setOutputSize(boundingBox.size());
                    ts.setBaseImage(textureImage[className].original, textureImage[className].mask);
                    ts.generateTexture();
                    ts.getMaskImage(texture);
                    
                }
                setTexture(texture, boundingBox);
            }
        }
    }
}

void SettingsTexture::getImageWithTexture(cv::Mat outImage) const
{
    mapImageWithTexture.copyTo(outImage);
}

void SettingsTexture::saveMapWithTexture(const std::string& fileName)
{
    if (colorMultiply > 1)
    {
        for (size_t i{}; i < mapImageWithTexture.rows; ++i)
        {
            for (size_t j{}; j < mapImageWithTexture.cols; ++j)
            {
                mapImageWithTexture.at<uchar>(i, j) *= colorMultiply;
            }
        }
    }
    std::string mapImageName{ fileName + ".png" };
    cv::imwrite(mapImageName, mapImageWithTexture);
    std::string mapLegendName{ fileName + ".txt" };
    std::ofstream fileWithClasse{ mapLegendName, std::ios::out | std::ios::trunc };
    if (fileWithClasse.is_open())
    {
        for (auto& n : classes)
        {
            fileWithClasse << n.second * colorMultiply << '\t';
            fileWithClasse << n.first << '\n';
        }
        fileWithClasse.close();
    }
}

void SettingsTexture::updateTextureImage()
{
    for (size_t i{}; i < quantityMainClasses; ++i)
    {
        std::string textureName{ getClassName(i) };
        std::string textureImageName{ getRandomTexture(getClassName(i)) };
        if (!textureImageName.empty())
        {
            std::string textureMaskName{ textureImageName + "_mask.png" };
            textureImageName += ".png";

            textureImage[textureName].mask = cv::imread(textureDirectory + "/" + textureName + "/" + textureMaskName, 0);
            textureImage[textureName].original = cv::imread(textureDirectory + "/" + textureName + "/" + textureImageName);
            repaintImage(textureImage[textureName].mask, classes, subClasses);
        }
    }
}

