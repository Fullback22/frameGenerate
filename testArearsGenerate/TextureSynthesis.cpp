#include <iostream>

#include "TextureSynthesis.h"

void TextureSynthesis::computeQuantityBloks()
{
    quatityBloks_.width = std::ceil((outputImageSize_.width - overlapWidth_) / static_cast<float>(blockSize_.width - overlapWidth_));
    quatityBloks_.height = std::ceil((outputImageSize_.height - overlapHeigth_) / static_cast<float>(blockSize_.height - overlapHeigth_));
}

void TextureSynthesis::setOutImageParams()
{
    int imageWidth{ (quatityBloks_.width * blockSize_.width) - (quatityBloks_.width - 1) * overlapWidth_ };
    int imageHeigth{ (quatityBloks_.height * blockSize_.height) - (quatityBloks_.height - 1) * overlapHeigth_ };
    outTextureImage_ = cv::Mat{ imageHeigth, imageWidth, CV_32FC3, cv::Scalar(0.0,0.0,0.0) };
    outMaskImage_ = cv::Mat{ imageHeigth, imageWidth, CV_8UC1, cv::Scalar(0) };
}

void TextureSynthesis::randomPatch()
{
    std::random_device rd{};
    std::mt19937 generator{ rd() };

    std::uniform_int_distribution<> disX{ 0, baseImage_.cols - blockSize_.width };
    std::uniform_int_distribution<> disY{ 0, baseImage_.rows - blockSize_.height };
    int x{ disX(generator) };
    int y{ disY(generator) };

    cv::Rect roi(x, y, blockSize_.width, blockSize_.height);
    part_ = baseImage_(roi);
    maskPart_ = baseMaskImage_(roi);
}

void TextureSynthesis::randomBestPatch(size_t const y, size_t const x)
{
    int rows{ baseImage_.rows - blockSize_.height };
    int cols{ baseImage_.cols - blockSize_.width };
    cv::Mat errors{ rows, cols, CV_32FC1, cv::Scalar{0.0} };
    float minError{ static_cast<float>(pow(blockSize_.area(),2)) };
    size_t iMin{};
    size_t jMin{};

    for (size_t i{}; i < baseImage_.rows - blockSize_.height; ++i)
    {
        for (size_t j{}; j < baseImage_.cols - blockSize_.width; ++j)
        {
            cv::Rect roi(j, i, blockSize_.width, blockSize_.height);
            cv::Mat testPath{ baseImage_(roi) };
            float error{ L2OverlapDiff(testPath, y, x) };
            if (minError > error)
            {
                minError = error;
                iMin = i;
                jMin = j;
            }
            errors.at<float>(i, j) = error;
        }
    }
    part_ = cv::Mat{ blockSize_, CV_32FC3 };
    maskPart_ = cv::Mat{ blockSize_, CV_8UC1 };
    for (size_t i{ iMin }, i2{}; i < iMin + blockSize_.height; ++i, ++i2)
    {
        for (size_t j{ jMin }, j2{}; j < jMin + blockSize_.width; ++j, ++j2)
        {
            part_.at<cv::Vec3f>(i2, j2) = baseImage_.at<cv::Vec3f>(i, j);
            maskPart_.at<uchar>(i2, j2) = baseMaskImage_.at<uchar>(i, j);
        }
    }
}

float TextureSynthesis::L2OverlapDiff(const cv::Mat& testPath, size_t const y, size_t const x)
{
    float outError{ 0.0 };
    if (x > 0)
    {
        for (size_t i{}, _y{y}; i < testPath.rows; ++i, ++_y)
        {
            for (size_t j{}, _x{ x }; j < overlapWidth_; ++j, ++_x)
            {
                const cv::Vec3f& d1{ testPath.at<cv::Vec3f>(i,j) };
                const cv::Vec3f& d2{ outTextureImage_.at<cv::Vec3f>(_y, _x) };
                for (size_t k{}; k < 3; ++k)
                {
                    outError += pow(d1[k] - d2[k], 2.0);
                }

            }
        }
    }
    if (y > 0)
    {
        for (size_t i{}, _y{ y }; i < overlapHeigth_; ++i, ++_y)
        {
            for (size_t j{}, _x{ x }; j < testPath.cols; ++j, ++_x)
            {
                const cv::Vec3f& d1{ testPath.at<cv::Vec3f>(i,j) };
                const cv::Vec3f& d2{ outTextureImage_.at<cv::Vec3f>(_y,_x) };
                for (size_t k{}; k < 3; ++k)
                {
                    outError += pow(d1[k] - d2[k], 2.0);
                }
            }
        }
    }
    if (x > 0 && y > 0)
    {
        for (size_t i{}, _y{y}; i < overlapHeigth_; ++i, ++_y)
        {
            for (size_t j{}, _x{ x }; j < overlapWidth_; ++j, ++_x)
            {
                const cv::Vec3f& d1{ testPath.at<cv::Vec3f>(i,j) };
                const cv::Vec3f& d2{ outTextureImage_.at<cv::Vec3f>(_y,_x) };
                for (size_t k{}; k < 3; ++k)
                {
                    outError -= pow(d1[k] - d2[k], 2.0);
                }
            }
        }
    }
    return outError;
}

void TextureSynthesis::minCutPatch(size_t const y, size_t const x)
{
    int dy{ part_.rows };
    int dx{ part_.cols };

    cv::Mat minCut{ part_.size(), CV_8UC1, cv::Scalar{0} };

    if (x > 0)
    {
        cv::Mat leftL2{ part_.rows, overlapWidth_, CV_32FC1, cv::Scalar{0.0} };

        for (size_t i{}, _y{ y }; i < part_.rows; ++i, ++_y)
        {
            for (size_t j{}, _x{ x }; j < overlapWidth_; ++j, ++_x)
            {
                const cv::Vec3f& d1{ part_.at<cv::Vec3f>(i, j) };
                const cv::Vec3f& d2{ outTextureImage_.at<cv::Vec3f>(_y, _x) };
                float outError{};
                for (size_t k{}; k < 3; ++k)
                {
                    outError += pow(d1[k] - d2[k], 2.0);
                }
                leftL2.at<float>(i, j) = outError;
            }
        }
        std::vector<int> part;

        minCutPath(leftL2, part);
        for (size_t i{}; i < part.size(); ++i)
        {
            for (size_t j{}; j < part[i]; ++j)
            {
                minCut.at<uchar>(i, j) = 1;
            }
        }
    }
    if (y > 0)
    {
        cv::Mat upL2{ overlapHeigth_, part_.cols, CV_32FC1, cv::Scalar{0.0} };
        for (size_t i{}, _y{ y }; i < overlapHeigth_; ++i, ++_y)
        {
            for (size_t j{}, _x{ x }; j < part_.cols; ++j, ++_x)
            {
                const cv::Vec3f& d1{ part_.at<cv::Vec3f>(i,j) };
                const cv::Vec3f& d2{ outTextureImage_.at<cv::Vec3f>(_y, _x) };
                float outError{};
                for (size_t k{}; k < 3; ++k)
                {
                    outError += pow(d1[k] - d2[k], 2.0);
                }
                upL2.at<float>(i, j) = outError;
            }
        }
        std::vector<int> part;
        upL2 = upL2.t();
        minCutPath(upL2, part);
        for (size_t i{}; i < part.size(); ++i)
        {
            for (size_t j{}; j < part[i]; ++j)
            {
                minCut.at<uchar>(i, j) = 1;
            }
        }
    }

    for (size_t i{}, _y{ y }; i < part_.rows; ++i, ++_y)
    {
        for (size_t j{}, _x{ x }; j < part_.cols; ++j, ++_x)
        {
            if (minCut.at<uchar>(i, j) == 1)
            {
                part_.at<cv::Vec3f>(i, j) = outTextureImage_.at<cv::Vec3f>(_y, _x);
                maskPart_.at<uchar>(i, j) = outMaskImage_.at<uchar>(_y, _x);
            }
        }
    }
}

void TextureSynthesis::minCutPath(cv::Mat& errors, std::vector<int>& path)
{
    std::map<int, float> errorsFirst;
    for (int i{}; i < errors.cols; ++i)
    {
        errorsFirst[i] = errors.at<float>(0, i);
    }
    std::vector<std::map<float, std::vector<int>>> pq(errorsFirst.size());

    for (size_t i{}; i < errorsFirst.size(); ++i)
    {
        pq[i][errorsFirst[i]].push_back(i);
    }
    std::make_heap(pq.begin(), pq.end(), std::greater<>{});
    std::set<std::pair<int, int>> seen;

    while (true)
    {
        std::pop_heap(pq.begin(), pq.end(), std::greater<>{});
        std::map<float, std::vector<int>> buf{ pq.back() };

        float error;
        for (const auto& n : buf)
        {
            error = n.first;
            path = n.second;
        }
        pq.pop_back();
        int curDepth = path.size();
        int curentIndex = path[path.size() - 1];
        if (curDepth == errors.rows)
        {
            return;
        }

        for (int delta{ -1 }; delta <= 1; ++delta)
        {
            int nextIndex = curentIndex + delta;
            if (0 <= nextIndex && nextIndex < errors.cols)
            {
                std::pair<int, int> test{ std::make_pair(curDepth, nextIndex) };
                if (seen.find(test) == seen.end())
                {
                    float cumError{ error + errors.at<float>(curDepth, nextIndex) };
                    std::vector<int> toPq{ path };
                    toPq.push_back(nextIndex);
                    pq.push_back({ {cumError, toPq} });
                    std::push_heap(pq.begin(), pq.end(), std::greater<>{});
                    seen.insert(test);
                }
            }
        }
    }
}

TextureSynthesis::TextureSynthesis()
{
}

TextureSynthesis::TextureSynthesis(const std::string& baseImageName, const std::string& baseMaskImageName, const cv::Size& outputSize, const cv::Size& blockSize)
{
    setBaseImage(baseImageName, baseMaskImageName);
    outputImageSize_ = outputSize;
    blockSize_ = blockSize;
    computeQuantityBloks();
    setOutImageParams();
}

void TextureSynthesis::setBaseImage(const std::string& baseImageName, const std::string& baseMaskImageName)
{
    baseImage_ = cv::imread(baseImageName);
    baseMaskImage_ = cv::imread(baseMaskImageName, 0);

    baseImage_.convertTo(baseImage_, CV_32FC3);
    for (size_t i{}; i < baseImage_.rows; ++i)
    {
        for (size_t j{}; j < baseImage_.cols; ++j)
        {
            baseImage_.at<cv::Vec3f>(i, j) /= 255.0;
        }
    }
}

void TextureSynthesis::setOutputSize(const cv::Size& outputSize)
{
    outputImageSize_ = outputSize;
    computeQuantityBloks();
}

void TextureSynthesis::setBlockSize(const cv::Size& blockSize)
{
    blockSize_ = blockSize;
    overlapWidth_ = blockSize_.width / overlapWidthCoefficient_;
    overlapHeigth_ = blockSize_.height / overlapHeigthCefficient_;
    computeQuantityBloks();
}

void TextureSynthesis::generateTexture()
{
    setOutImageParams();
    for (size_t i{}; i < quatityBloks_.height; ++i)
    {
        for (size_t j{}; j < quatityBloks_.width; ++j)
        {
            size_t x{ j * (blockSize_.width - overlapWidth_) };
            size_t y{ i * (blockSize_.height - overlapHeigth_) };

            if (i == 0 && j == 0)
            {
                randomPatch();
            }
            else
            {
                randomBestPatch(y, x);
                minCutPatch(y, x);
            }
            for (size_t k{ y }, k2{}; k < y + blockSize_.height; ++k, ++k2)
            {
                for (size_t l{ x }, l2{}; l < x + blockSize_.width; ++l, ++l2)
                {
                    outTextureImage_.at<cv::Vec3f>(k, l) = part_.at<cv::Vec3f>(k2, l2);
                    outMaskImage_.at<uchar>(k, l) = maskPart_.at<uchar>(k2, l2);
                }
            }
        }
    }
    cv::Rect roi{ 0,0,outputImageSize_.width, outputImageSize_.height };
    outTextureImage_ = outTextureImage_(roi);
    outMaskImage_ = outMaskImage_(roi);
}

void TextureSynthesis::getTextureImage(cv::Mat& texture) const
{
    outTextureImage_.copyTo(texture);
}

void TextureSynthesis::getMaskImage(cv::Mat& mask) const
{
    outMaskImage_.copyTo(mask);
}

void TextureSynthesis::setBaseImage(const cv::Mat& baseImage, const cv::Mat& baseMaskImage)
{
    baseImage.copyTo(baseImage_);
    baseMaskImage.copyTo(baseMaskImage_);

    baseImage_.convertTo(baseImage_, CV_32FC3);
    for (size_t i{}; i < baseImage_.rows; ++i)
    {
        for (size_t j{}; j < baseImage_.cols; ++j)
        {
            baseImage_.at<cv::Vec3f>(i, j) /= 255.0;
        }
    }
}
