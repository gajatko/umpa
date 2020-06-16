#pragma once
#include "Casino.h"
#include "Bot.h"
#include "Human.h"
using std::pair;

class Game
{
	Casino* casino;

	void status() const;
	bool checkContinuePlaying(Player* player) const;
	void round(pair<vector<Bot*>, Human*> players) const;
	pair<vector<Bot*>, Human*> addPlayers() const;
	Choice makeChoice() const;
	void randomizeFinishHandler(int number, Color color) const;
	pair<vector<Bot*>, Human*> extractPlayers(vector<Player*> players) const;

public:
	void start() const;
	Game(Casino* casino);
};