#pragma once
#include <vector>
#include<random>

class ProbabilityOfPosition
{
	std::vector<std::vector<double>> probability{};
	int stepResetToZeroOffset{ 0 };
	int stepUpdateOffset{ 0 };
	int offsetProbability{ 0 };
	double probabilityOfOffset{ 0.0 };
	std::uniform_int_distribution<> offsetDist{};
	std::random_device rd{};
	std::mt19937 gen;
	bool checkProbability(double const probobility);
public:
	ProbabilityOfPosition();
	ProbabilityOfPosition(int const lowerOffsetValue, 
							int const upperOffsetValue, 
							int const lowerUpdateOffsetValue, 
							int const upperUpdateOffsetValue, 
							double const probabilityOfOffset, 
							int const multiplicityResetToZeroOffset = 1);

	ProbabilityOfPosition(const ProbabilityOfPosition &drop);
	void setStepUpdateOffset(int const lowerUpdateOffsetValue, int const upperUpdateOffsetValue);
	void setStepResetToZeroOffset(int const multiplicityResetToZeroOffset = 1);
	void setProbability(std::vector<std::vector<double>>* const newPropobility);
	void setProbabilityOfOffset(double const newProbabilityOfOfsset);
	std::vector<double> getProbolity(int const index, int const indexForOffset);
};

