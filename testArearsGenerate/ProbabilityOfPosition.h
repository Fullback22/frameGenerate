#pragma once
#include <vector>
#include<random>

class ProbabilityOfPosition
{
protected:
	std::vector<std::vector<double>> probabilitiesOfClasses_{};
	size_t offsetOfProbability_{};
	double probabilityOfOffset_{};
	
	size_t updateingOffsetStep_{};
	size_t stepResetToZeroOffset_{};
	
	std::uniform_int_distribution<> offsetDist_{};
	std::mt19937 generator_{};

	bool allowOffsetChange();
public:
	ProbabilityOfPosition();
	ProbabilityOfPosition(int const lowerOffsetValue,
						int const upperOffsetValue,
						int const lowerOffsetUpdateValue,
						int const upperOffsetUpdateValue,
						double const probabilityOfOffset,
						int const multiplicityResetToZeroOffset = 1);
	ProbabilityOfPosition(const ProbabilityOfPosition &drop);

	void setProbability(const std::vector<std::vector<double>>& newPropobility);
	void setStepUpdateOffset(int const lowerUpdateOffsetValue, int const upperUpdateOffsetValue);
	void setStepResetToZeroOffset(int const multiplicityResetToZeroOffset = 1);
	void setProbabilityOfOffset(double const newProbabilityOfOfsset);
	
	void updateOffsetOfProbability(const size_t index);
	std::vector<double> getProbolity(size_t const index) const;
};