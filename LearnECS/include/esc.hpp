#pragma once  // 防止头文件被多次包含

#include <vector>        // 包含vector容器
#include <algorithm>     // 包含算法库
#include <cassert>       // 包含断言宏
#include <unordered_map> // 包含哈希表容器
#include "sparse_set.hpp" // 包含稀疏集合头文件

#define assertm(msg,expr) assert(((void)msg,(expr))) // 定义带消息的断言宏

//class Commonds;
//class Queryer;
//class Resources;
namespace ecs  // 定义ecs命名空间
{
    using ComponentID = uint32_t;  // 组件ID类型别名
    using Entity = uint32_t;       // 实体类型别名
    using SystemID = uint32_t;     // 系统ID类型别名

    struct Component final{};
    struct Resource final{};

    // IndexGetter类用于获取组件类型的唯一ID
    template<typename Category>
    class IndexGetter final
    {
    public:
        // 模板方法，为每种类型T生成唯一的ComponentID
        template<typename T>
        static uint32_t Get()
        {
            static uint32_t id = curIdx_++;  // 静态变量保证每种类型只有一个ID
            return id;
        }
    private:
        inline static uint32_t curIdx_ = 0;  // 当前ID计数器
    };

    // IDGenerator结构体用于生成唯一ID
    template<typename T, typename = std::enable_if<std::is_integral_v<T>>>
    struct IDGenerator final
    {
    public:
        // 生成新的ID
        static ComponentID Gen()
        {
            return curIdx_++;  // 返回当前ID并递增
        }
    private:
        inline static ComponentID curIdx_ = 0;  // ID计数器
    };

    using EntityGenerator = IDGenerator<Entity>; // 实体生成器别名


    // World类，ECS系统的核心，管理所有实体和组件
    class World final
    {
    public:
        friend class Commonds;  // 声明Commonds为友元类
        friend class Resources;
        friend class Queryer;

        using UpdateSystem = void (*)(Commonds&, Queryer, Resources);
        using StartupSystem = void (*)(Commonds&);
        using ComponentContainer = std::unordered_map<ComponentID, void*>;  // 组件容器类型别名

        World() = default;
        World(World&&) = delete;
        World& operator=(World&&) = delete;

        template<typename T>
        World& SetResource(T&& resource);

        World& AddStartupSystem(StartupSystem system)
        {
            startupSystems_.push_back(system);
            return *this;
        }

        World& AddSystem(UpdateSystem system)
        {
            updateSystems_.push_back(system);
            return *this;
        }

        inline void Startup();
        inline void Update();
        inline void Shutdown();

    private:
        // Pool结构体，用于管理组件对象池
        struct Pool final
        {
            std::vector<void*> instances;  // 存储所有实例的向量
            std::vector<void*> cache;      // 缓存已销毁的实例

            using CreateFunc = void* (*)();      // 创建函数指针类型
            using DestroyFunc = void (*)(void*); // 销毁函数指针类型

            CreateFunc create;   // 创建函数指针
            DestroyFunc destroy; // 销毁函数指针

            // 构造函数，初始化创建和销毁函数
            Pool(CreateFunc inCreate, DestroyFunc inDestroy)
                : create(inCreate), destroy(inDestroy)
            {
                assertm("you must give a non-null create or destory func", create && destroy);
            }

            // 析构函数，销毁所有实例
            ~Pool()
            {
                for (auto instance : instances)
                {
                    destroy(instance);
                }
            }

            // 创建新实例
            void* Create()
            {
                void* instance = nullptr;
                if (cache.empty())  // 如果缓存为空
                {
                    instance = create();  // 调用创建函数
                }
                else  // 如果缓存不为空
                {
                    instance = cache.back();  // 从缓存中取
                    cache.pop_back();
                }
                instances.push_back(instance);  // 添加到实例列表
                return instance;
            }

            // 销毁实例
            void Destroy(void* instance)
            {
                // 查找要销毁的实例
                if (auto it = std::find(instances.begin(), instances.end(), instance);
                    it != instances.end())
                {
                    cache.push_back(*it);           // 添加到缓存
                    std::swap(*it, instances.back()); // 与最后一个元素交换
                    instances.pop_back();           // 删除最后一个元素
                }
                else
                {
                    assertm("Invalid instance", false);  // 断言无效实例
                }
            }
        };

        // ComponentInfo结构体，包含组件池和实体集合
        struct ComponentInfo final
        {
            Pool pool;                           // 组件池
            SparseSet<Entity, 32> entitySet;      // 实体集合

            // 添加实体到集合
            void AddEntity(Entity entity)
            {
                entitySet.insert(entity);
            }

            // 从集合中移除实体
            void RemoveEntity(Entity entity)
            {
                entitySet.erase(entity);
            }

            // 构造函数，初始化组件池
            ComponentInfo(Pool::CreateFunc inCreate, Pool::DestroyFunc inDestroy)
                : pool(inCreate, inDestroy)
            {
            }

            // 默认构造函数
            ComponentInfo() :pool(nullptr, nullptr)
            {
            }

        };

        using ComponentPool = std::unordered_map<ComponentID, ComponentInfo>;  // 组件池类型别名
        ComponentPool componentMap_;  // 组件映射表

        std::unordered_map<Entity, ComponentContainer> entities_;  // 实体映射表

        struct ResourceInfo final
        {
            void* resource = nullptr;  // 资源实例
            using DestroyFunc = void (*)(void*); // 销毁函数指针类型

            DestroyFunc destroy;

            ResourceInfo(DestroyFunc inDestroy)
                :destroy(inDestroy)
            {
                assertm("you must give a non-null create or destory func", destroy);
            }

            ~ResourceInfo()
            {
                destroy(resource);
            }
        };
        std::unordered_map<ComponentID, ResourceInfo> resources_;

        std::vector<UpdateSystem> updateSystems_;
        std::vector<StartupSystem> startupSystems_;
    };

    // Commonds类，用于执行创建/销毁实体等命令
    class Commonds final
    {
    public:
        using DestroyFunc = void(*)(void*);
        using AssignFunc = std::function<void(void*)>;
        using CreateFunc = std::function<void*()>;
        struct ResourceDestroyInfo final
        {
            DestroyFunc destroy_;
            uint32_t index_;
            ResourceDestroyInfo(DestroyFunc destroy, uint32_t index)
                : destroy_(destroy)
                , index_(index)
            {
            }
        };

        struct ComponentSpawnInfo final
        {
            AssignFunc assign_;
            World::Pool::CreateFunc create_;
            World::Pool::DestroyFunc destroy_;
            ComponentID index_;
        };

        struct EntitySpawnInfo final
        {
            Entity entity_;
            std::vector<ComponentSpawnInfo> components;
            EntitySpawnInfo(Entity entity)
                : entity_(entity)
            {
            }
        };

        struct ResourceSpawnInfo final
        {
            CreateFunc create_;
            World::Pool::DestroyFunc destroy_;
            uint32_t index_;
        };
    public:
        // 构造函数，初始化World引用
        Commonds(World& world)
            : world_(world)
        {
        }

        // Spawn方法，创建带有指定组件的实体
        template<typename... ComponentTypes>
        Commonds& Spawn(ComponentTypes&& ... components)
        {
            SpawnAndReturn<ComponentTypes...>(std::forward<ComponentTypes>(components)...);  // 执行创建
            return *this;  // 返回自身以支持链式调用
        }

        template<typename... ComponentTypes>
        Entity SpawnAndReturn(ComponentTypes&& ... components)
        {
            EntitySpawnInfo info(EntityGenerator::Gen());
            doSpawn(info.entity_,info.components, std::forward<ComponentTypes>(components)...);  // 执行创建
            entitySpawnInfos_.push_back(info);
            return info.entity_;  // 返回新创建的实体ID
        }

        // Destroy方法，销毁指定实体
        Commonds Destroy(Entity entity)
        {
            destoryEntities_.push_back(entity);
            return *this;  // 返回自身以支持链式调用
        }


        template<typename T>
        Commonds& SetResource(T&& resource)
        {
            ResourceSpawnInfo ResourceInfo;
            ResourceInfo.index_ = IndexGetter<Resource>::Get<T>();
            ResourceInfo.create_ = [resource = std::forward<T>(resource)]()->void* {
                return new T(resource);
            };
            ResourceInfo.destroy_ = [](void* instance) {
                delete (T*)instance;
            };
            resourceSpawnInfos_.push_back(ResourceInfo);
            
            return *this;
        }

        template<typename T>
        Commonds& RemoveResource()
        {
            uint32_t index = IndexGetter<Resource>::Get<T>(); 
            resourceDestroyInfos_.push_back(ResourceDestroyInfo(
                [](void* instance) {delete (T*)instance; },
                index
            ));
            return *this;
        }

        void Execute()
        {
            for (auto info : resourceDestroyInfos_)
            {
                removeResource(info);
            }
            for (auto e : destoryEntities_)
            {
                destroyEntity(e);
            }

            for (auto spawnInfo : entitySpawnInfos_)
            {
                auto it = world_.entities_.emplace(spawnInfo.entity_, World::ComponentContainer{});
                auto& componentContainer = it.first->second;
                for (auto& componentInfo : spawnInfo.components)
                {
                    componentContainer[componentInfo.index_] = doSpawnWithoutType(spawnInfo.entity_, componentInfo);
                }
            }

            for (auto info : resourceSpawnInfos_)
            {
                setResource(info);
            }
        }
    private:
        // doSpawn方法，递归地为实体添加组件
        template<typename T, typename ... Remains>
        void doSpawn(Entity entity,std::vector<ComponentSpawnInfo>& ComponentSpawnInfos, T&& component, Remains&& ... remains)
        {
            ComponentSpawnInfo info;
            info.index_ = IndexGetter<Component>::Get<T>();  // 获取组件类型ID
            info.create_ = []()->void* {
                return new T(); 
            };          // 创建函数
            info.destroy_ = [](void* instance) {
                delete (T*)instance; 
            }; // 销毁函数

            info.assign_ = [=](void* instance) {
                //static auto com = std::forward<T>(component);
                *((T*)instance) = component;
            };
            ComponentSpawnInfos.push_back(info);
            // 如果还有剩余组件，递归处理
            if constexpr (sizeof...(Remains) != 0)
            {
                doSpawn<Remains...>(entity, ComponentSpawnInfos, std::forward<Remains>(remains)...);
            }
        }

        void* doSpawnWithoutType(Entity entity, ComponentSpawnInfo& info)
        {
            if (auto it = world_.componentMap_.find(info.index_); it == world_.componentMap_.end())
            {
                world_.componentMap_.emplace(info.index_, World::ComponentInfo(info.create_, info.destroy_));
            }
            World::ComponentInfo& componentInfo = world_.componentMap_[info.index_];
            void* elem = componentInfo.pool.Create();
            info.assign_(elem);
            componentInfo.AddEntity(entity);

            return elem;
        }

        // Destroy方法，销毁指定实体
        void destroyEntity(Entity entity)
        {
            // 查找实体
            if (auto it = world_.entities_.find(entity); it != world_.entities_.end())
            {
                // 遍历实体的所有组件
                for (auto& [index, component] : it->second)
                {
                    auto& componentInfo = world_.componentMap_[index];  // 获取组件信息
                    componentInfo.pool.Destroy(component);              // 销毁组件实例
                    componentInfo.RemoveEntity(entity);                 // 从实体集合中移除
                }
                world_.entities_.erase(it);  // 从实体映射表中删除实体
            }
        }

        void setResource(const ResourceSpawnInfo& Info)
        {
            auto index = Info.index_;
            if (auto it = world_.resources_.find(index); it != world_.resources_.end())
            {
                // 先释放原有资源
                if (it->second.resource) {
                    it->second.destroy(it->second.resource);
                }
                // 创建新资源
                it->second.resource = Info.create_();
            }
            else
            {
                auto NewIt = world_.resources_.emplace(index, World::ResourceInfo(
                    Info.destroy_
                ));

                // 使用传入的resource来创建资源，保持行为一致性
                NewIt.first->second.resource = Info.create_();
            }
        }

        void removeResource(const ResourceDestroyInfo& Info)
        {
            if (auto it = world_.resources_.find(Info.index_); it != world_.resources_.end())
            {
                Info.destroy_(it->second.resource);
                it->second.resource = nullptr;
            }
        }
    private:
        World& world_;  // World引用

        std::vector<Entity> destoryEntities_;

        std::vector<ResourceDestroyInfo> resourceDestroyInfos_;

        std::vector<EntitySpawnInfo> entitySpawnInfos_;

        std::vector<ResourceSpawnInfo> resourceSpawnInfos_;
    };

    class Resources {
        public:
            Resources(World& world): world_(world){}

            template<typename T>
            bool Has()
            {
                auto index = IndexGetter<Resource>::Get<T>();
                auto it = world_.resources_.find(index);
                return it != world_.resources_.end() && it->second.resource;
            }

            template<typename T>
            T& Get()
            {
                auto index = IndexGetter<Resource>::Get<T>();
                auto it = world_.resources_.find(index);
                return *((T*)it->second.resource);
            }
    private:
        World& world_;
    };

    class Queryer final
    { 
    public:
        Queryer(World& world)
            : world_(world)
        {
        }   

        template<typename ... ComponentTypes>
        std::vector<Entity> Query()
        {
            std::vector<Entity> entities;
            doQuery<ComponentTypes...>(entities);
            return entities;
        }

        template<typename T>
        bool Has(Entity entity)
        {
            auto it = world_.entities_.find(entity);
            auto Index = IndexGetter<Component>::Get<T>();
            return it != world_.entities_.end() && it->second.find(Index) != it->second.end();
        }

        template<typename T>
        T& Get(Entity entity)
        {
            auto it = world_.entities_.find(entity);
            auto Index = IndexGetter<Component>::Get<T>();
            return *((T*)it->second[Index]);
        }
    private:
        template<typename T, typename ... Remains>
        void doQuery(std::vector<Entity>& outEntities)
        {
            auto index = IndexGetter<Component>::Get<T>();
            auto it = world_.componentMap_.find(index);
            if (it == world_.componentMap_.end()) return;

            for (const auto& entity : it->second.entitySet)
            {
                if constexpr (sizeof...(Remains) == 0)
                {
                    outEntities.push_back(entity);
                }
                else
                {
                    if (doQueryRemains<Remains...>(entity))
                    {
                        outEntities.push_back(entity);
                    }
                }
            }
        }

        template<typename T, typename ... Remains>
        bool doQueryRemains(Entity entity)
        {
            auto index = IndexGetter<Component>::Get<T>();
            auto entityIt = world_.entities_.find(entity);
            if (entityIt == world_.entities_.end() || entityIt->second.find(index) == entityIt->second.end())
            {
                return false;
            }

            if constexpr (sizeof...(Remains) == 0)
            {
                return true;
            }
            else
            {
                return doQueryRemains<Remains...>(entity);
            }
        }

    private:

        World& world_;
    };

    void World::Startup()
    {
        std::vector<Commonds> commondsList;
        for (auto sys : startupSystems_)
        {
            Commonds commonds{ *this };
            sys(commonds);
            commondsList.emplace_back(commonds);
        }

        for (auto& commond : commondsList)
        {
            commond.Execute();
        }
    }
    
    void World::Update()
    {

        std::vector<Commonds> commondsList;
        for (auto sys : updateSystems_)
        {
            Commonds commonds{ *this };
            sys(Commonds{ *this }, Queryer{ *this }, Resources{ *this });
            commondsList.emplace_back(commonds);
        }

        for (auto& commond : commondsList)
        {
            commond.Execute();
        }
    }

    void World::Shutdown()
    {
        entities_.clear();
        componentMap_.clear();
        resources_.clear();
    }

    template<typename T>
    inline World& World::SetResource(T&& resource)
    {
        Commonds commonds{ *this };
        commonds.SetResource(std::forward<T>(resource));
        return *this;
    }
}