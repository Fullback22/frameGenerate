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
                testureBlock.width = channelJson.at("textureBlockWidth").get<int>();
                testureBlock.height = channelJson.at("textureBlockHeigth").get<int>();


            }
            catch (...)
            {
                std::cout << "ERROR: JSON READ" << std::endl;
                paramFile.close();
            }
            paramFile.close();
        }
        setClassesTexture();
    }
}

void SettingsTexture::setMainClasses(const json& channelJson)
{
    std::vector<std::string> buferForClassNames;
    buferForClassNames = channelJson.at("mainClasses").get<std::vector<std::string>>();
    int color{ 20 };
    int const colorStep{ 20 };
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
        std::string format{ ".txt" };
        std::ifstream fileWithClasse{ mainClass.first + format };
        std::map<std::string, int> textureClasses;
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
                    textureClasses[className] = classColor;
                }
            }
            fileWithClasse.close();
            std::string maskFormat{ "Mask.png" };
            textureImage[mainClass.first] = cv::imread(mainClass.first + maskFormat, 0);
            repaintTextureImage(mainClass.first, textureClasses);
        }
        for (auto& n : textureClasses)
            classes.insert(n);
    }
}

void SettingsTexture::repaintTextureImage(const std::string& textureName, std::map<std::string, int>& textureClasses)
{
    std::vector<std::pair<int, int>> cls(classes.size());
    int color{ static_cast<int>(classes.size()) };

    int i{  };
    for (auto& n : textureClasses)
    {
        cls[i].first = n.second;
        cls[i].second = color;
        n.second = color;
        ++i;
        ++color;
    }

    const std::pair<int, int>* activPair{ &cls[0] };
    for (size_t i{}; i < textureImage[textureName].rows; ++i)
    {
        for (size_t j{}; j < textureImage[textureName].cols; ++j)
        {
            if (activPair->first != textureImage[textureName].at<uchar>(i, j))
            {
                for (auto& p : cls)
                {
                    if (p.first == textureImage[textureName].at<uchar>(i, j))
                    {
                        activPair = &p;
                        break;
                    }
                }
            }
            textureImage[textureName].at<uchar>(i, j) = activPair->second;
        }
    }
}

