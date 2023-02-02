#include "ProbabilityOfPosition.h"

ProbabilityOfPosition::ProbabilityOfPosition():
	offsetDist{0,0},
	gen{ rd() }
{
}

ProbabilityOfPosition::ProbabilityOfPosition(int const lowerOffsetValue, 
											int const upperOffsetValue, 
											int const lowerUpdateOffsetValue, 
											int const upperUpdateOffsetValue, 
											double const probabilityOfOffset_, 
											int const multiplicityResetToZeroOffset):
	offsetDist{lowerOffsetValue, upperOffsetValue},
	probabilityOfOffset{ probabilityOfOffset_ },
	gen{ rd() }
{
	setStepUpdateOffset(lowerUpdateOffsetValue, upperUpdateOffsetValue);
	setStepResetToZeroOffset(multiplicityResetToZeroOffset);
}

bool ProbabilityOfPosition::checkProbability(double const probobility)
{
	std::uniform_real_distribution<> checkDist{0.0, 1.0};
	if (checkDist(gen) < probobility)
		return true;
	else
		return false;
}

void ProbabilityOfPosition::setStepUpdateOffset(int const lowerUpdateOffsetValue, int const upperUpdateOffsetValue)
{
	std::uniform_int_distribution<> stepUpdateOffsetDist{ lowerUpdateOffsetValue, upperUpdateOffsetValue };
	stepUpdateOffset = stepUpdateOffsetDist(gen);
}

std::vector<double> ProbabilityOfPosition::getProbolity(int const index, int const indexForOffset)
{
	int indexInPropobility{ index + offsetProbability };
	if (indexInPropobility >= probability[0].size())
		indexInPropobility = probability[0].size() - 1;
	std::vector<double> outProbobility{};
	for (auto classProbobility : probability)
	{
		outProbobility.push_back(classProbobility[indexInPropobility]);
	}
	return outProbobility;
}

void ProbabilityOfPosition::setProbabilityOfOffset(double const newProbabilityOfOfsset)
{
	probabilityOfOffset = newProbabilityOfOfsset;
}


ProbabilityOfPosition::ProbabilityOfPosition(const ProbabilityOfPosition& drop):
	stepResetToZeroOffset{ drop.stepResetToZeroOffset },
	stepUpdateOffset{drop.stepUpdateOffset},
	offsetDist{drop.offsetDist},
	offsetProbability{drop.offsetProbability },
	probabilityOfOffset{drop.probabilityOfOffset},
	gen{ drop.gen }
{
	probability.assign(drop.probability.begin(), drop.probability.end());
}

void ProbabilityOfPosition::setProbability(std::vector<std::vector<double>>* const newProbobility)
{
	probability.assign(newProbobility->begin(), newProbobility->end());
}

void ProbabilityOfPosition::setStepResetToZeroOffset(int const multiplicityResetToZeroOffset)
{
	stepResetToZeroOffset = stepUpdateOffset * multiplicityResetToZeroOffset;
}
