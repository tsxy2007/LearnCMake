#pragma once  // 防止头文件被多次包含

#include <vector>        // 包含vector容器
#include <algorithm>     // 包含算法库
#include <cassert>       // 包含断言宏
#include <unordered_map> // 包含哈希表容器
#include "sparse_set.hpp" // 包含稀疏集合头文件

#define assertm(msg,expr) assert(((void)msg,(expr))) // 定义带消息的断言宏

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
        using ComponentContainer = std::unordered_map<ComponentID, void*>;  // 组件容器类型别名

        World() = default;
        World(World&&) = delete;
        World& operator=(World&&) = delete;
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
    };

    // Commonds类，用于执行创建/销毁实体等命令
    class Commonds final
    {
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
            Entity entity = EntityGenerator::Gen();  // 生成新实体ID
            doSpawn(entity, std::forward<ComponentTypes>(components)...);  // 执行创建
            return *this;  // 返回自身以支持链式调用
        }

        // Destroy方法，销毁指定实体
        Commonds Destroy(Entity entity)
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
            return *this;  // 返回自身以支持链式调用
        }


        template<typename T>
        Commonds& SetResource(T&& resource)
        {
            auto index = IndexGetter<Resource>::Get<T>();
            if (auto it = world_.resources_.find(index); it != world_.resources_.end())
            {
                // 先释放原有资源
                if (it->second.resource) {
                    it->second.destroy(it->second.resource);
                }
                // 创建新资源
                it->second.resource = new T(std::forward<T>(resource));
            }
            else
            {
                auto NewIt = world_.resources_.emplace(index, World::ResourceInfo(
                    [](void* instance) {delete (T*)instance; }
                ));

                // 使用传入的resource来创建资源，保持行为一致性
                NewIt.first->second.resource = new T(std::forward<T>(resource));
            }
            return *this;
        }

        template<typename T>
        Commonds& RemoveResource()
        {
            if (auto it = world_.resources_.find(IndexGetter<Resource>::Get<T>()); it != world_.resources_.end())
            {
                delete (T*)it->second.resource;
                it->second.resource = nullptr;
            }
            return *this;
        }
    private:
        // doSpawn方法，递归地为实体添加组件
        template<typename T, typename ... Remains>
        void doSpawn(Entity entity, T&& component, Remains&& ... remains)
        {
            auto Index = IndexGetter<Component>::Get<T>();  // 获取组件类型ID
            // 如果组件类型尚未注册
            if (auto it = world_.componentMap_.find(Index); it == world_.componentMap_.end())
            {
                // 注册新组件类型
                world_.componentMap_.emplace(Index, World::ComponentInfo(
                    []()->void* {return new T(); },           // 创建函数
                    [](void* instance) {delete (T*)instance; } // 销毁函数
                ));
            }
            auto& componentInfo = world_.componentMap_[Index];  // 获取组件信息
            void* elem = componentInfo.pool.Create();           // 创建组件实例
            *((T*)elem) = std::forward<T>(component);           // 赋值组件数据
            componentInfo.AddEntity(entity);                    // 添加实体到集合

            // 在实体映射表中添加组件
            auto it = world_.entities_.emplace(entity, World::ComponentContainer{});
            it.first->second[Index] = elem;

            // 如果还有剩余组件，递归处理
            if constexpr (sizeof...(Remains) != 0)
            {
                doSpawn<Remains...>(entity, std::forward<Remains>(remains)...);
            }
        }
    private:
        World& world_;  // World引用
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
}