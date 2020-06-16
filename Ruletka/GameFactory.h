#pragma once
#include "Casino.h"

class GameFactory
{
private: 
	int getIntLine(std::wifstream& file);

public:
	void save(Casino* casssino);
	Casino* load();
	Casino* create();
};
