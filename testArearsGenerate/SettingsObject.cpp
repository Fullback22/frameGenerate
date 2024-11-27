#include "SettingsObject.h"

void SettingsObject::setMainClasses(const json& channelJson)
{
    std::vector<std::string> buferForClassNames;
    buferForClassNames = channelJson.at("mainClasses").get<std::vector<std::string>>();
    int color{ 0 };
    int const colorStep{ 1 };
    for (const auto& className : buferForClassNames)
    {
        params.mainClasses[className] = color;
        color += colorStep;
    }
}

void SettingsObject::loadImage()
{
    std::string imageDirectory{ "object" };
    std::string dirName{ imageDirectory + "/" + "air" };
    for (const auto& image : fs::directory_iterator(dirName))
    {
        airImage.push_back(cv::imread(image.path().string(), 0));
    }

    dirName = imageDirectory + "/" + "land";
    for (const auto& image : fs::directory_iterator(dirName))
    {
        landImage.push_back(cv::imread(image.path().string(), 0));
    }
}

void SettingsObject::resizeImage(cv::Mat& image)
{
    updateImageHeigth();
    cv::Size newSize{ (image.cols * objectHeigth) / image.rows,objectHeigth };
    cv::resize(image, image, newSize);
    cv::threshold(image, image, 0, 255, cv::THRESH_OTSU);
}

void SettingsObject::updateImageHeigth()
{
    std::uniform_int_distribution<> objectHeigthDistr{ params.minObjectHeight, params.maxObjectHeight };
    std::random_device rd{};
    std::mt19937 generator{ rd() };

    objectHeigth = objectHeigthDistr(generator);
}

cv::Point SettingsObject::getStartPopintForObject(const cv::Mat backgroundImage, const cv::Mat object)
{
    std::vector<int> landColor;
    for (const auto& landClass: params.landClasses)
    {
        landColor.push_back(params.mainClasses[landClass]);
    }

    for (;;)
    {
        std::uniform_int_distribution<> xDistr{ 0, backgroundImage.cols - object.cols - 1 };
        std::uniform_int_distribution<> yDistr{ 0 , backgroundImage.rows - object.rows - 1 };
        std::random_device rd{};
        std::mt19937 generator{ rd() };

        int x = xDistr(generator);
        int y = yDistr(generator);
        
        bool isGoodPoint{ true };
        for (size_t i{}; i < object.rows && isGoodPoint; ++i)
        {
            int testPixelValue{ backgroundImage.at<uchar>(y + object.rows, x + i) };
            if (std::find(landColor.begin(), landColor.end(), testPixelValue) == landColor.end())
                isGoodPoint = false;
        }
        if (isGoodPoint)
            return cv::Point(x, y);
    }
}

SettingsObject::SettingsObject(const std::string& fileName)
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
                json channelJson = modelParametrs.at("OBJECT_PARAMETRS");
                setMainClasses(channelJson);
                params.landClasses = channelJson.at("landClasses").get<std::vector<std::string>>();
                params.airClasses = channelJson.at("airClasses").get<std::vector<std::string>>();
                params.maxQuantityLandObject = channelJson.at("maxQuantityLandObject").get<int>();
                params.maxQuantityAirObject = channelJson.at("maxQuantityAirObject").get<int>();
                params.minQuantityLandObject = channelJson.at("minQuantityLandObject").get<int>();
                params.minQuantityAirObject = channelJson.at("minQuantityAirObject").get<int>();
                params.maxObjectHeight = channelJson.at("maxObjectHeight").get<int>();
                params.minObjectHeight = channelJson.at("minObjectHeight").get<int>();
            }
            catch (...)
            {
                std::cout << "ERROR: JSON READ" << std::endl;
                paramFile.close();
            }
            paramFile.close();
        }
    }
    quantityAirObject = params.minQuantityAirObject;
    quantityLandObject = params.minQuantityLandObject;
    objectHeigth = params.minObjectHeight;
    loadImage();
}

void SettingsObject::updateQuantityObjectOnImage()
{
    std::uniform_int_distribution<> quantityAirOjectDistr{ params.minQuantityAirObject, params.maxQuantityAirObject };
    std::uniform_int_distribution<> quantityLandObjectDistr{ params.minQuantityLandObject, params.maxQuantityLandObject };
    std::uniform_int_distribution<> objectHeigthDistr{ params.minObjectHeight, params.maxObjectHeight };
    std::random_device rd{};
    std::mt19937 generator{ rd() };

    quantityAirObject = quantityAirOjectDistr(generator);
    quantityLandObject = quantityLandObjectDistr(generator);
    objectHeigth = objectHeigthDistr(generator);
}

void SettingsObject::setObject(cv::Mat& inOutImage)
{
    std::uniform_int_distribution<> numberLandObjectDistr{ 0, static_cast<int>(landImage.size() - 1) };
    std::random_device rd{};
    std::mt19937 generator{ rd() };
    int color{ params.mainClasses["land"] };
    
    for (int i{}; i < quantityLandObject; ++i)
    {
        int imageNumber{ numberLandObjectDistr(generator) };
        cv::Mat object{};
        landImage[imageNumber].copyTo(object);
        resizeImage(object);
        cv::Point startPoint{ getStartPopintForObject(inOutImage, object) };
        for (size_t i{}; i < object.rows; ++i)
        {
            for (size_t j{}; j < object.cols; ++j)
            {
                if (object.at<uchar>(i, j) == 0)
                    inOutImage.at<uchar>(i + startPoint.y, j + startPoint.x) = color;
            }
        }
        writeObjectCoordinate(startPoint.x, startPoint.y, object.size(), static_cast<int>(OBJECT_TYPE::LAND));
    }

    std::uniform_int_distribution<> numberAirObjectDistr{ 0, static_cast<int>(landImage.size() - 1) };
    color = params.mainClasses["air"];
    for (int i{}; i < quantityAirObject; ++i)
    {
        int imageNumber{ numberAirObjectDistr(generator) };
        cv::Mat object{};
        landImage[imageNumber].copyTo(object);
        resizeImage(object);
        cv::Point startPoint{ getStartPopintForObject(inOutImage, object) };
        for (size_t i{}; i < object.rows; ++i)
        {
            for (size_t j{}; j < object.cols; ++j)
            {
                if (object.at<uchar>(i, j) == 0)
                    inOutImage.at<uchar>(i + startPoint.y, j + startPoint.x) = color;
            }
        }
        writeObjectCoordinate(startPoint.x, startPoint.y, object.size(), static_cast<int>(OBJECT_TYPE::AIR));
    }
}

void SettingsObject::writeObjectCoordinate(const int x, const int y, const cv::Size& size, const int type) const
{
    std::ofstream outFile{ fileName + "objects.txt", std::ios::out | std::ios::app };
    if (outFile.is_open())
    {
        outFile << x << '\t' << y << '\t' << size.width << '\t' << size.height << '\t' << type << '\n';
        outFile.close();
    }
}

void SettingsObject::setFileName(const std::string newFileName)
{
    fileName = newFileName;
}
