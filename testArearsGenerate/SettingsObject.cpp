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
}

void SettingsObject::updateSettings()
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
