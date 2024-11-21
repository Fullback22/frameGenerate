#include "AreaParametr.h"

AreaParametr::AreaParametr(const std::string& fileName)
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
                json channelJson = modelParametrs.at("AREA_PARAMETRS");
                quantityImage = channelJson.at("quantityImage").get<size_t>();
                startNumber = channelJson.at("startNumber").get<size_t>();
                quantityClasses = channelJson.at("quantityClasses").get<size_t>();
                callSize.width = channelJson.at("callWidth").get<int>();
                callSize.height = channelJson.at("callHeigth").get<int>();
                weigthMapSize.width = channelJson.at("weigthMapWidth").get<int>();
                weigthMapSize.height = channelJson.at("weigthMapHeigth").get<int>();
                weigthsForWeigthMap = channelJson.at("weigthMap").get<std::vector<double>>();
                landAirProportion = channelJson.at("landAirProportion").get<double>();
                transitionMap.resize(quantityClasses);
                transitionMap = channelJson.at("transitionMap").get<std::vector<std::vector<int>>>();
                probabilityOfPosition = channelJson.at("probabilityOfPosition").get<std::vector<std::string>>();
                multiplicityResetToZeroOffset = channelJson.at("multiplicityResetToZeroOffset").get<int>();
                lowerOffsetUpdateValue = channelJson.at("lowerOffsetUpdateValue").get<int>();
                upperOffsetUpdateValue = channelJson.at("upperOffsetUpdateValue").get<int>();
                lowerOffsetValue = channelJson.at("lowerOffsetValue").get<int>();
                upperOffsetValue = channelJson.at("upperOffsetValue").get<int>();
				probabilityOfOffset = channelJson.at("probabilityOfOffset").get<float>();
            }
            catch (...)
            {
                std::cout << "ERROR: JSON READ" << std::endl;
                paramFile.close();
            }
            paramFile.close();
        }
    }
}

void AreaParametr::setProbabilityOfPosition(std::vector<std::vector<double>>& outProbability, double const positionOffset)
{
	MySigmoid initProbabilityOfPositionMainClasses{ positionOffset, 0.3 };
	std::vector<std::vector<std::string>> probabilityParametrs(probabilityOfPosition.size());

	for (size_t i{}, j{}; i < probabilityOfPosition.size(); ++i, ++j)
	{
		parsString(probabilityOfPosition[i], probabilityParametrs[j]);
		if (std::atoi(probabilityParametrs[j][0].c_str()) >= quantityClasses)
		{
			probabilityParametrs.erase(probabilityParametrs.begin() + j);
			--j;
		}
	}
	probabilityOfPosition.clear();

	for (int j{ 0 }; j < outProbability[0].size(); ++j)
	{
		for (int i{}; i < probabilityParametrs.size(); ++i)
		{
			int classIndex{ std::atoi(probabilityParametrs[i][0].c_str()) };
			if (probabilityParametrs[i][1] == "s")
			{
				outProbability[classIndex][j] = initProbabilityOfPositionMainClasses.getValue(j);
				for (size_t s{ 2 }; s < probabilityParametrs[i].size(); s += 2)
				{
					switch (probabilityParametrs[i][s][0])
					{
					case('+'):
					{
						outProbability[classIndex][j] += std::atof(probabilityParametrs[i][s + 1].c_str());
						outProbability[classIndex][j] = abs(outProbability[classIndex][j]);
						break;
					}
					case('*'):
					{
						outProbability[classIndex][j] *= std::atof(probabilityParametrs[i][s + 1].c_str());
						break;
					}
					default:
						break;
					}
				}
			}
			else if (probabilityParametrs[i][1] == "1")
			{
				int upperBorder{ static_cast<int>(std::atof(probabilityParametrs[i][2].c_str()) * outProbability[0].size()) };
				int downBorder{ static_cast<int>(std::atof(probabilityParametrs[i][3].c_str()) * outProbability[0].size()) };
				if (j > upperBorder && j <= downBorder)
				{
					outProbability[classIndex][j] = std::atof(probabilityParametrs[i][4].c_str());
				}
			}
			else
			{
				std::cout << "WARRNING: incorrect position probability parameter" << std::endl;
			}
		}
	}
}

void AreaParametr::parsString(const std::string& input, std::vector<std::string>& output, const std::string& delimiter)
{
	size_t pos_start{};
	size_t pos_end{};
	size_t delim_len{ delimiter.size() };
	std::string token;

	while ((pos_end = input.find(delimiter, pos_start)) != std::string::npos)
	{
		token = input.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		output.push_back(token);
	}

	output.push_back(input.substr(pos_start));
}
