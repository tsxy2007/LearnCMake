#include <cstdio>
#include <vector>
#include "include/esc.hpp"
#include <iostream>

struct Name final {
    std::string name;
};
struct ID final {
    int id;
};

struct Timer final {
    int time = 123;
};

void StartupSystemFunc(ecs::Commonds commonds) 
{
    commonds.Spawn(Name{ "person1" })
        .Spawn(Name{ "person2" }, ID{ 1 })
        .Spawn(Timer{ 111 })
        .Spawn(ID{ 2 })
        .SetResource(Timer{ 222 });
}

void ECSUpdateNameSystem(ecs::Commonds commonds, ecs::Queryer queryer, ecs::Resources resources)
{
    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<Name>()) {
        std::cout << queryer.Get<Name>(entity).name << std::endl;
    }
}

void ECSUpdateNameAndIDSystem(ecs::Commonds commonds, ecs::Queryer queryer, ecs::Resources resources)
{
    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<Name, ID>())
    {
        std::cout << queryer.Get<Name>(entity).name << " , " << queryer.Get<ID>(entity).id << std::endl;
    }
}

void ECSUpdateIDSystem(ecs::Commonds commonds, ecs::Queryer queryer, ecs::Resources resources)
{
    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<ID>()) 
    {
        std::cout << queryer.Get<ID>(entity).id << std::endl;
    }
}

void ECSUpdateTimerSystem(ecs::Commonds& commonds, ecs::Queryer& queryer, ecs::Resources& resources)
{
    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<Timer>())
    {
        std::cout << queryer.Get<Timer>(entity).time << std::endl;
    }
}

int main() 
{
    using namespace ecs;
    ecs::World world;
    world.AddStartupSystem(StartupSystemFunc)
        .AddSystem(ECSUpdateNameSystem)
        .AddSystem(ECSUpdateNameAndIDSystem)
        .AddSystem(ECSUpdateIDSystem);
    
    world.Startup();

    world.Update();

    world.Shutdown();

    printf("Hello, Learn ECS!\n");
    return 0;
}