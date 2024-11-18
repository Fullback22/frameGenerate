#pragma once

#include <random>
#include <map>
#include <vector>
#include <set>

#include <opencv2/highgui/highgui.hpp>

class TextureSynthesis
{
	cv::Mat baseImage_{};
	cv::Mat baseMaskImage_{};
	cv::Mat outMaskImage_{};
	cv::Mat outTextureImage_{};

	cv::Size outputImageSize_{ 300,300 };
	cv::Size blockSize_{ 50,50 };
	

	int overlapWidthCoefficient_{ 6 };
	int overlapHeigthCefficient_{ 6 };

	int overlapWidth_{ blockSize_.width / overlapWidthCoefficient_ };
	int overlapHeigth_{ blockSize_.height / overlapHeigthCefficient_ };

	cv::Size quatityBloks_{};

	cv::Mat part_{};
	cv::Mat maskPart_{};

	void computeQuantityBloks();
	void setOutImageParams();

	void randomPatch();
	void randomBestPatch(size_t const y, size_t const x);
	float L2OverlapDiff(const cv::Mat& testPath, size_t const y, size_t const x);
	void minCutPatch(size_t const y, size_t const x);
	void minCutPath(cv::Mat& errors, std::vector<int>& path);
public:
	TextureSynthesis();
	TextureSynthesis(const std::string& baseImageName, const std::string& baseMaskImageName, const cv::Size& outputSize = cv::Size{ 300, 300 }, const cv::Size& blockSize = cv::Size{ 50,50 });

	void setBaseImage(const std::string& baseImageName, const std::string& baseMaskImageName);
	void setBaseImage(const cv::Mat& baseImage, const cv::Mat& baseMaskImage);
	void setOutputSize(const cv::Size& outputSize);
	void setBlockSize(const cv::Size& blockSize);

	void generateTexture();
	void getTextureImage(cv::Mat& texture) const;
	void getMaskImage(cv::Mat& mask) const;
};

