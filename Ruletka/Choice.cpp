#include "Choice.h"

Choice::Choice(int cash, vector<int> numbers, vector<Color> colors)
{
	for (auto i = numbers.begin(); i < numbers.end(); i++) {
		if (*i < 1 || *i > 35) {
			throw ILLEGAL_NUMBER;
		}
	}
	this->cash = cash;
	this->numbers = numbers;
	this->colors = colors;
}

vector<int> Choice::getNumbers()
{
	return numbers;
}

vector<Color> Choice::getColors()
{
	return colors;
}

int Choice::getCash()
{
	return this->cash;
}

Choice::Choice()
{
	this->cash = 0;
}
