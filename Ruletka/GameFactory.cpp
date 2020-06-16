#include "GameFactory.h"
#include "Bot.h"
#include "Human.h"

#include <fstream>
#include <sstream>

using std::endl;

void GameFactory::save(Casino* casino)
{
    auto players = casino->getPlayers();
    std::wofstream myfile;
	myfile.open("save.txt");
    myfile.imbue(std::locale("pl_PL.utf8"));
    myfile << players.size() << endl;
    for (auto p = players.begin(); p < players.end(); p++)
    {
        Player* player = *p;
        bool isBot = player->isBot(); 

        myfile << player->nick << endl;
        myfile << (isBot ? 1 : 0) << endl;
        myfile << player->getCash() << endl;
    }
	myfile.close();
}

int GameFactory::getIntLine(std::wifstream& file) {
	int x;
	std::wstringstream str;
	wstring line;
	getline(file, line);
	str << line;
	str >> x;
	return x;
}


Casino* GameFactory::load()
{
    auto casino = new Casino();
    
    std::wifstream myfile;
	myfile.open("save.txt");
    myfile.imbue(std::locale("pl_PL.utf8"));
    int playerCount = getIntLine(myfile);
    
    for (int i = 0; i < playerCount; i++) 
    {
        wstring nick;
        int isBotInt;
        int cash;


        getline(myfile, nick);
        isBotInt = getIntLine(myfile);
        cash = getIntLine(myfile);

        bool isBot = isBotInt == 1;
        Player* p;
        if (isBot) {
            p = new Bot(nick, cash);
        }
        else {
            p = new Human(nick, cash);
        }

        casino->addPlayer(p);
    }
	myfile.close();

    return casino;

}

Casino* GameFactory::create()
{
    return new Casino();
}
