#pragma once
#include <string>
#include "Choice.h"

using std::wstring;


class Player
{
private:
	Choice* choice;

protected:
	int cash;
	void setChoicePtr(Choice* choice);
	Player(); 

public:
	std::wstring nick;

	~Player();
	Choice* getChoice();
	void giveCash(int amount);
	void takeCash(int amount);
	Player(wstring nick, int cash);
	int getCash();

	virtual bool isBot() = 0;

	static const int NOT_ENOUGH_CASH = 1;
};