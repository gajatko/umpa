#include "Bot.h"
#include <sstream>

void Bot::makeRandomChoice()
{
	vector<Color> colors;
	vector<int> numbers;

	bool betColor = rand() % 2 == 1;
	int possibleLossCount = 0;

	if (betColor)
	{
		int c = rand() % 2 + 1;
		colors.push_back(c == 1 ? Color::Black : Color::Red);
		possibleLossCount++;
	}

	int l = rand() % 35 + 1;
	numbers.push_back(l);
	possibleLossCount++;

	bool canBet = this->getCash() / possibleLossCount > 0;

	int cash = canBet ? (rand() % (this->getCash() / possibleLossCount)) + 1 : 0;
	setChoicePtr(new Choice(cash, numbers, colors));
}

Bot::Bot(int cash)
{
	std::wstringstream botNameStr;
	botNameStr << L"Bot " << rand() % 1000;
	this->nick = botNameStr.str();
	this->cash = cash;
}

Bot::Bot(wstring nick, int cash)
{ 
	this->nick = nick;
	this->cash = cash;
}

bool Bot::isBot()
{
	return true;
}
