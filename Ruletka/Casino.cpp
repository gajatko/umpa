#include "Casino.h"
#include <iostream>

using std::wcout;
using std::endl;

Casino::Casino()
{
}

Casino::~Casino()
{
    for (auto p = players.begin(); p < players.end(); p++)
    {
        delete* p;
    }
    players.clear();
}

void Casino::nextRound()
{ 
    /*
    1. Gracz obstawił x na kolor
       0) traci x
       1) dostaje x
    2. Gracz obstawił x na liczbę
       0) traci x
       1) dostaje 34x
   */

    int winNumber = rand() % 35 + 1;
    Color winColor = winNumber % 2 ? Color::Black : Color::Red;

    if (randomizeFinishHandler) {
        randomizeFinishHandler(winNumber, winColor);
    }

    for (int i = 0; i < players.size(); i++) {
        Player* player = players[i];
		Choice* choice = player->getChoice();
        if (choice == nullptr) {
            continue;
        }
        int cash = choice->getCash();

		if (!choice->getColors().empty()) {
            bool success = choice->getColors()[0] == winColor;
			if (success) {
				player->giveCash(cash);
			}
			else {
				player->takeCash(cash);
			}
		}

        auto numbers = choice->getNumbers();

		for (auto n = numbers.begin(); n < numbers.end(); n++) {
			if (*n == winNumber) {
                int win = 34 * cash;
				player->giveCash(win);
			}
			else {
                int loss = cash;
				player->takeCash(loss);
			}
		}
        if (player->getCash() <= 0) {
            players.erase(players.begin() + i);
            i--;
        }
	}
}

vector<Player*> Casino::getPlayers() const
{
    return this->players;
}

void Casino::addPlayer(Player* player)
{
    players.push_back(player);
}

void Casino::removePlayer()
{
}
