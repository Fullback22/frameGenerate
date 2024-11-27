#include "ArearsGenerate.h"
#include "AreaParametr.h"
#include "SettingsTexture.h"
#include "TextureSynthesis.h"
#include "SettingsObject.h"

#include <Windows.h>
#include <vector>
#include <list>
#include <map>
#include <algorithm>

int main()
{
	AreaParametr areaParametrs{ "modelParametrs.json" };
    SettingsTexture texture("modelParametrs.json");
    SettingsObject objects{ "modelParametrs.json" };

	std::vector<cv::Size> standartImageSize{ {640, 480}, {800,600}, {960, 540}, {1024, 600}, {1280, 720}, {1280, 1024}, {1600, 900}, {1920, 1080}, {2048,1080} };
	//std::vector<cv::Size> standartImageSize{ {640, 480} };
	int quantityOfSize{ static_cast<int>(standartImageSize.size()) };

	std::uniform_int_distribution<> imageSizeDistr{ 0, quantityOfSize - 1 };
	std::random_device rd{};
	std::mt19937 generator{ rd() };

    for (size_t i{ }; i < areaParametrs.quantityImage; ++i)
    {
        std::cout << i+1 << " / " << areaParametrs.quantityImage << std::endl;
        const int numberOfImageSize{ imageSizeDistr(generator) };
        const cv::Size& imageSize{ standartImageSize[numberOfImageSize] };

        int landAirBorder{ static_cast<int>(imageSize.height * areaParametrs.landAirProportion) };
        int landAirSigma = landAirBorder * 0.1;

        std::uniform_int_distribution<> initDist{ landAirBorder - landAirSigma, landAirBorder + landAirSigma };
        double positionOffset{ static_cast<double>(initDist(generator)) };


        std::vector<std::vector<double>> probabilityOfPosition(areaParametrs.quantityClasses, std::vector<double>(imageSize.height));

        areaParametrs.setProbabilityOfPosition(probabilityOfPosition, positionOffset);

        ProbabilityOfPosition probobility{ areaParametrs.lowerOffsetValue, 
            areaParametrs.upperOffsetValue, 
            imageSize.width / areaParametrs.upperOffsetUpdateValue, 
            imageSize.width / areaParametrs.lowerOffsetUpdateValue, 
            areaParametrs.probabilityOfOffset,
            areaParametrs.multiplicityResetToZeroOffset };
        probobility.setProbability(probabilityOfPosition);

        ArearsGenerate myModel{ imageSize };
        myModel.setClassesParametrs(areaParametrs.quantityClasses, areaParametrs.callSize, areaParametrs.weigthMapSize, areaParametrs.weigthsForWeigthMap);
        myModel.setProbabilityOfPosition(probobility);
        myModel.setTrasitionMap(areaParametrs.transitionMap);


        cv::Mat imageWithMainClasses(myModel.generateImage());
        
        std::string mapName{ "myModel_areas/myModel_" + std::to_string(i + areaParametrs.startNumber) };
        
        objects.setFileName(mapName);
        objects.updateQuantityObjectOnImage();
        objects.setObject(imageWithMainClasses);

        texture.setMapImage(imageWithMainClasses);
        texture.updateTextureImage();
        texture.addTextureToMapImage();

        texture.saveMapWithTexture(mapName);
    }
	return 0;
}
