#include "MainMenu.h"
#include "GameFactory.h"
#include "Game.h"
#include <iostream>
using namespace std;

using std::wcout;

void MainMenu::show()
{
	GameFactory fucktory;

	setlocale(LC_ALL, "pl_PL");
	// wyśw. menu nowa gra, wczytaj
	bool choosing = true;

	Casino* casino = nullptr;
	while (choosing) {
		int wybor;
		wcout << u8"Wybierz opcję.\n1. Nowa gra\n2. Wczytaj gre" << std::endl;
		cin >> wybor;
		if (wybor == 1)
		{
			casino = fucktory.create();
			choosing = false;
		}
		else if (wybor == 2)
		{
			casino = fucktory.load();
			choosing = false;
		}
		else {
			wcout << L"Zły wybór." << std::endl;
		}
	}
	Game game(casino);
	game.start();

	auto players = casino->getPlayers(); 
	auto human = std::find_if(players.begin(), players.end(), [](Player* p) { return !p->isBot();  });
	if (human != players.end()) {
		wcout << L"Czy chcesz zapisać stan gry (1=tak)? ";
		int saveGame;
		wcin >> saveGame;
		if (saveGame == 1) {
			fucktory.save(casino);
		}
	}
}
