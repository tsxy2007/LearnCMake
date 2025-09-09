#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include "sparse_set.hpp"

#define assertm(msg,expr) assert(((void)msg,(expr)))

namespace ecs 
{
    using ComponentID = uint32_t;
    using EntityID = uint32_t;
    using SystemID = uint32_t;

    class IndexGetter final
    {
    public:
        template<typename T>
        static ComponentID GetID(const std::string& name)
        {
            static ComponentID id = curIdx_ ++;
            return id;
        }
    private:
        inline static ComponentID curIdx_ = 0;
    };

	class World final
	{
	public:
        friend class Commands;
        using ComponentContainer = std::unordered_map<ComponentID, void*>;
	private:
		struct Pool final
		{ 
			std::vector<void*> instances;
			std::vector<void*> cache;

			using CreateFunc = void* (*)();
			using DestroyFunc = void (*)(void*);

            CreateFunc create;
            DestroyFunc destroy;

			Pool(CreateFunc inCreate, DestroyFunc inDestroy)
				: create(inCreate), destroy(inDestroy)
			{
			}	
            ~Pool()
            {
                for (auto instance : instances)
                {
                    destroy(instance);
                }
            }
            void* Create()
            {
                void* instance = nullptr;
                if (cache.empty())
                {
                    instance = create();
                }
                else
                {
                    instance = cache.back();
                    cache.pop_back();
                }
                instances.push_back(instance);
                return instance;
            }

            void Destroy(void* instance)
            {
                if (auto it = std::find(instances.begin(), instances.end(), instance); 
                    it != instances.end())
                {
                    cache.push_back(*it);
                    std::swap(*it, instances.back());
                    instances.pop_back();
                }
                else
                {
                    assertm("Invalid instance", false);
                }
            }
		};

        struct ComponentInfo final
        {
            Pool pool;
            SparseSet<EntityID,128> entitySet;

            void AddEntity(EntityID entity)
            {
                entitySet.insert(entity);
            }

            void RemoveEntity(EntityID entity)
            {
                entitySet.erase(entity);
            }
        };

        using ComponentPool = std::unordered_map<ComponentID, ComponentInfo>;
        ComponentPool ComponentMap_;

        std::unordered_map<EntityID, ComponentContainer> entities_;
	};

    class Commonds final
    {
    public:
    private:
    };
}