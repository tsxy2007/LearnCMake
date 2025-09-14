#include <cstdio>
#include <vector>
#include "include/esc.hpp"

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
    ecs::World world;
    ecs::Commonds commonds(world);
    commonds.Spawn(Name{ "person1" })
            .Spawn(Name{ "person1" },ID{ 1 })
            .Spawn(Timer{ 111 })
            .SetResource(Timer{222});
    printf("Hello, Learn ECS!\n");
    commonds.RemoveResource<Timer>();
    return 0;
}