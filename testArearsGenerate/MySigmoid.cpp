#include "MySigmoid.h"

MySigmoid::MySigmoid(double offset_, double coolness_):
	offset{offset_},
	coolness{coolness_}
{
}

double MySigmoid::getValue(double const x) const
{
	double toExp{ (x - offset) * coolness };
	return 1.0/(1.0 + exp(toExp));
}

void MySigmoid::setOffset(double const newOffset)
{
	offset = newOffset;
}

void MySigmoid::setCoolness(double const newCoolness)
{
	coolness = newCoolness;
}
