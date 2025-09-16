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

int main() 
{
    using namespace ecs;
    ecs::World world;
    ecs::Commonds commonds(world);
    commonds.Spawn(Name{ "person1" })
            .Spawn(Name{ "person2" },ID{ 1 })
            .Spawn(Timer{ 111 })
            .Spawn(ID{ 2 })
            .SetResource(Timer{222});

    Queryer queryer(world);
    
    for (auto entity : queryer.Query<Name>()) {
        std::cout << queryer.Get<Name>(entity).name << std::endl;
    }

    std::cout<<"-----------------------------"<<std::endl;
    for (auto entity : queryer.Query<Name,ID>()) {
        std::cout << queryer.Get<Name>(entity).name << " , " << queryer.Get<ID>(entity).id << std::endl;
    }

    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<ID>()) {
        std::cout << queryer.Get<ID>(entity).id << std::endl;
    }

    std::cout << "-----------------------------" << std::endl;
    for (auto entity : queryer.Query<Timer>()) {
        std::cout << queryer.Get<Timer>(entity).time << std::endl;
    }

    std::cout << "-----------------------------" << std::endl;
    Resources resources(world);
    std::cout << resources.Get<Timer>().time << std::endl;

    printf("Hello, Learn ECS!\n");
    commonds.RemoveResource<Timer>();
    return 0;
}