#include "ProbabilityOfPosition.h"

ProbabilityOfPosition::ProbabilityOfPosition()
{
	std::random_device randomDevice{};
	generator_.seed(randomDevice());
}

ProbabilityOfPosition::ProbabilityOfPosition(int const lowerOffsetValue,
											int const upperOffsetValue,
											int const lowerOffsetUpdateValue,
											int const upperOffsetUpdateValue,
											double const probabilityOfOffset,
											int const multiplicityResetToZeroOffset):
probabilityOfOffset_{ probabilityOfOffset }
{
	std::random_device randomDevice{};
	generator_.seed(randomDevice());
	setStepUpdateOffset(lowerOffsetUpdateValue, upperOffsetUpdateValue);
	setStepResetToZeroOffset(multiplicityResetToZeroOffset);
}

void ProbabilityOfPosition::setStepUpdateOffset(int const lowerUpdateOffsetValue, int const upperUpdateOffsetValue)
{
	std::uniform_int_distribution<> stepUpdateOffsetDist{ lowerUpdateOffsetValue, upperUpdateOffsetValue };
	updateingOffsetStep_ = stepUpdateOffsetDist(generator_);
}

void ProbabilityOfPosition::setStepResetToZeroOffset(int const multiplicityResetToZeroOffset)
{
	stepResetToZeroOffset_ = updateingOffsetStep_ * multiplicityResetToZeroOffset;
}

ProbabilityOfPosition::ProbabilityOfPosition(const ProbabilityOfPosition& drop) :
	probabilitiesOfClasses_{ drop.probabilitiesOfClasses_.begin(), drop.probabilitiesOfClasses_.end() },
	offsetOfProbability_{ drop.offsetOfProbability_ },
	probabilityOfOffset_{ drop.probabilityOfOffset_ },
	updateingOffsetStep_{ drop.updateingOffsetStep_ },
	stepResetToZeroOffset_{ drop.stepResetToZeroOffset_ },
	offsetDist_{ drop.offsetDist_ },
	generator_{ drop.generator_ }
{
}

std::vector<double> ProbabilityOfPosition::getProbolity(size_t const index) const
{
	size_t propobilityIndex{ index + offsetOfProbability_ };
	if (propobilityIndex >= probabilitiesOfClasses_[0].size())
		propobilityIndex = probabilitiesOfClasses_[0].size() - 1;

	std::vector<double> outProbobility(probabilitiesOfClasses_.size());
	for (size_t i{};i<probabilitiesOfClasses_.size();++i)
	{
		outProbobility[i] = probabilitiesOfClasses_[i][propobilityIndex];
	}
	return outProbobility;
}

void ProbabilityOfPosition::setProbabilityOfOffset(double const newProbabilityOfOfsset)
{
	probabilityOfOffset_ = newProbabilityOfOfsset;
}

void ProbabilityOfPosition::setProbability(const std::vector<std::vector<double>>& newProbobility)
{
	probabilitiesOfClasses_.assign(newProbobility.begin(), newProbobility.end());
}

void ProbabilityOfPosition::updateOffsetOfProbability(const size_t index)
{
	if (index % stepResetToZeroOffset_ == 0)
	{
		offsetOfProbability_ = 0;
	}
	if (index % updateingOffsetStep_ == 0)
	{
		int newOffset{ offsetDist_(generator_) };
		if (allowOffsetChange())
			offsetOfProbability_ = newOffset;
	}
}

bool ProbabilityOfPosition::allowOffsetChange()
{
	std::uniform_real_distribution<> checkDist{ 0.0, 1.0 };
	if (checkDist(generator_) < probabilityOfOffset_)
		return true;
	else
		return false;
}
