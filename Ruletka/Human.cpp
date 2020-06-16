#include "Human.h"

using std::vector;

Human::Human(std::wstring nick, int cash) //konstruktor
{
	this->nick = nick;
	this->cash = cash; 
}

void Human::setChoice(int cash, vector<int> numbers, vector<Color> colors) //ustawianie wyboru gracza
{
 	setChoicePtr(new Choice(cash, numbers, colors));
}

bool Human::isBot()
{
	return false;
}
