#include <clocale>
#include <iostream>
#include "windows.h"
#include "MainMenu.h" 

int main()
{
    // Żeby były polskie znaki diak.
    // 1. ustawić locale:
    std::setlocale(LC_ALL, "pl_PL");
    SetConsoleOutputCP(CP_UTF8);
    // 2. Każdy plik w którym się pojawiają takie znaki zapisać z kodowaniem "UTF-8 without signature"
    //       (File->Save As->Save With Encoding)
    // 3. W ustawieniach konfiguracji (ang. "Ruletka Properties") Configuration Properties > C/C++ > Command Line
    //    dodać opcję lini poleceń "/utf-8".

    std::wcout.imbue(std::locale("pl_PL.utf8"));
    std::wcin.imbue(std::locale("pl_PL.utf8"));

    MainMenu menu;
    menu.show(); 

}