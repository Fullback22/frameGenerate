#pragma once
#include<math.h>

class MySigmoid
{
	double offset{};
	double coolness{};
public:
	MySigmoid(double offset = 0, double coolness = 1.0);
	double getValue(double const x) const;
	void setOffset(double const newOffset);
	void setCoolness(double const newCoolness);
};

