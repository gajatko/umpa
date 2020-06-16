#include "Player.h"

Choice* Player::getChoice()
{
    return choice;
}

void Player::giveCash(int amount)
{
    this->cash = cash + amount;
}

void Player::takeCash(int amount)
{
    this->cash = cash - amount;
}

void Player::setChoicePtr(Choice* choice)
{
    if (this->choice != nullptr) {
        delete this->choice; 
    }
    int possibleLossCount = choice->getColors().size() + choice->getNumbers().size();
    if (choice->getCash()*possibleLossCount > cash) {
        throw NOT_ENOUGH_CASH;
    }
    this->choice = choice;
}

Player::Player()
{
    this->cash = 0; 
    this->choice = nullptr;
}

Player::~Player()
{
    if (this->choice != nullptr) {
        delete choice;
        choice = nullptr;
    }
}

Player::Player(std::wstring nick, int cash)
{
    this->cash = cash; 
    this->nick = nick;
    this->choice = nullptr;
}

int Player::getCash()
{
    return cash;
}
