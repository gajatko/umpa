#pragma once
#include <vector>
#include <string>
#include "Number.h"
#include "Color.h"
#include "Player.h"

class Human : public Player
{
public:
	Human(std::wstring nick, int cash);


	void setChoice(int cash, std::vector<int> numbers, std::vector<Color> colors);
	bool isBot();
};