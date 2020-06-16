#pragma once
#include "Player.h"
#include <vector>
#include <functional>

using std::vector;

class Casino
{
private:
	vector<Player*> players;
public:
	std::function<void(int, Color)> randomizeFinishHandler;


	Casino();
	~Casino();

	void nextRound();
	vector<Player*> getPlayers() const;
	void addPlayer(Player* player);
	void removePlayer();
};