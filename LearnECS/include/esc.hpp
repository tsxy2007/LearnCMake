#pragma once  // ��ֹͷ�ļ�����ΰ���

#include <vector>        // ����vector����
#include <algorithm>     // �����㷨��
#include <cassert>       // �������Ժ�
#include <unordered_map> // ������ϣ������
#include "sparse_set.hpp" // ����ϡ�輯��ͷ�ļ�

#define assertm(msg,expr) assert(((void)msg,(expr))) // �������Ϣ�Ķ��Ժ�

//class Commonds;
//class Queryer;
//class Resources;
namespace ecs  // ����ecs�����ռ�
{
    using ComponentID = uint32_t;  // ���ID���ͱ���
    using Entity = uint32_t;       // ʵ�����ͱ���
    using SystemID = uint32_t;     // ϵͳID���ͱ���

    struct Component final{};
    struct Resource final{};

    // IndexGetter�����ڻ�ȡ������͵�ΨһID
    template<typename Category>
    class IndexGetter final
    {
    public:
        // ģ�巽����Ϊÿ������T����Ψһ��ComponentID
        template<typename T>
        static uint32_t Get()
        {
            static uint32_t id = curIdx_++;  // ��̬������֤ÿ������ֻ��һ��ID
            return id;
        }
    private:
        inline static uint32_t curIdx_ = 0;  // ��ǰID������
    };

    // IDGenerator�ṹ����������ΨһID
    template<typename T, typename = std::enable_if<std::is_integral_v<T>>>
    struct IDGenerator final
    {
    public:
        // �����µ�ID
        static ComponentID Gen()
        {
            return curIdx_++;  // ���ص�ǰID������
        }
    private:
        inline static ComponentID curIdx_ = 0;  // ID������
    };

    using EntityGenerator = IDGenerator<Entity>; // ʵ������������


    // World�࣬ECSϵͳ�ĺ��ģ���������ʵ������
    class World final
    {
    public:
        friend class Commonds;  // ����CommondsΪ��Ԫ��
        friend class Resources;
        friend class Queryer;

        using UpdateSystem = void (*)(Commonds&, Queryer, Resources);
        using StartupSystem = void (*)(Commonds&);
        using ComponentContainer = std::unordered_map<ComponentID, void*>;  // ����������ͱ���

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
        // Pool�ṹ�壬���ڹ�����������
        struct Pool final
        {
            std::vector<void*> instances;  // �洢����ʵ��������
            std::vector<void*> cache;      // ���������ٵ�ʵ��

            using CreateFunc = void* (*)();      // ��������ָ������
            using DestroyFunc = void (*)(void*); // ���ٺ���ָ������

            CreateFunc create;   // ��������ָ��
            DestroyFunc destroy; // ���ٺ���ָ��

            // ���캯������ʼ�����������ٺ���
            Pool(CreateFunc inCreate, DestroyFunc inDestroy)
                : create(inCreate), destroy(inDestroy)
            {
                assertm("you must give a non-null create or destory func", create && destroy);
            }

            // ������������������ʵ��
            ~Pool()
            {
                for (auto instance : instances)
                {
                    destroy(instance);
                }
            }

            // ������ʵ��
            void* Create()
            {
                void* instance = nullptr;
                if (cache.empty())  // �������Ϊ��
                {
                    instance = create();  // ���ô�������
                }
                else  // ������治Ϊ��
                {
                    instance = cache.back();  // �ӻ�����ȡ
                    cache.pop_back();
                }
                instances.push_back(instance);  // ��ӵ�ʵ���б�
                return instance;
            }

            // ����ʵ��
            void Destroy(void* instance)
            {
                // ����Ҫ���ٵ�ʵ��
                if (auto it = std::find(instances.begin(), instances.end(), instance);
                    it != instances.end())
                {
                    cache.push_back(*it);           // ��ӵ�����
                    std::swap(*it, instances.back()); // �����һ��Ԫ�ؽ���
                    instances.pop_back();           // ɾ�����һ��Ԫ��
                }
                else
                {
                    assertm("Invalid instance", false);  // ������Чʵ��
                }
            }
        };

        // ComponentInfo�ṹ�壬��������غ�ʵ�弯��
        struct ComponentInfo final
        {
            Pool pool;                           // �����
            SparseSet<Entity, 32> entitySet;      // ʵ�弯��

            // ���ʵ�嵽����
            void AddEntity(Entity entity)
            {
                entitySet.insert(entity);
            }

            // �Ӽ������Ƴ�ʵ��
            void RemoveEntity(Entity entity)
            {
                entitySet.erase(entity);
            }

            // ���캯������ʼ�������
            ComponentInfo(Pool::CreateFunc inCreate, Pool::DestroyFunc inDestroy)
                : pool(inCreate, inDestroy)
            {
            }

            // Ĭ�Ϲ��캯��
            ComponentInfo() :pool(nullptr, nullptr)
            {
            }

        };

        using ComponentPool = std::unordered_map<ComponentID, ComponentInfo>;  // ��������ͱ���
        ComponentPool componentMap_;  // ���ӳ���

        std::unordered_map<Entity, ComponentContainer> entities_;  // ʵ��ӳ���

        struct ResourceInfo final
        {
            void* resource = nullptr;  // ��Դʵ��
            using DestroyFunc = void (*)(void*); // ���ٺ���ָ������

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

    // Commonds�࣬����ִ�д���/����ʵ�������
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
        // ���캯������ʼ��World����
        Commonds(World& world)
            : world_(world)
        {
        }

        // Spawn��������������ָ�������ʵ��
        template<typename... ComponentTypes>
        Commonds& Spawn(ComponentTypes&& ... components)
        {
            SpawnAndReturn<ComponentTypes...>(std::forward<ComponentTypes>(components)...);  // ִ�д���
            return *this;  // ����������֧����ʽ����
        }

        template<typename... ComponentTypes>
        Entity SpawnAndReturn(ComponentTypes&& ... components)
        {
            EntitySpawnInfo info(EntityGenerator::Gen());
            doSpawn(info.entity_,info.components, std::forward<ComponentTypes>(components)...);  // ִ�д���
            entitySpawnInfos_.push_back(info);
            return info.entity_;  // �����´�����ʵ��ID
        }

        // Destroy����������ָ��ʵ��
        Commonds Destroy(Entity entity)
        {
            destoryEntities_.push_back(entity);
            return *this;  // ����������֧����ʽ����
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
        // doSpawn�������ݹ��Ϊʵ��������
        template<typename T, typename ... Remains>
        void doSpawn(Entity entity,std::vector<ComponentSpawnInfo>& ComponentSpawnInfos, T&& component, Remains&& ... remains)
        {
            ComponentSpawnInfo info;
            info.index_ = IndexGetter<Component>::Get<T>();  // ��ȡ�������ID
            info.create_ = []()->void* {
                return new T(); 
            };          // ��������
            info.destroy_ = [](void* instance) {
                delete (T*)instance; 
            }; // ���ٺ���

            info.assign_ = [=](void* instance) {
                //static auto com = std::forward<T>(component);
                *((T*)instance) = component;
            };
            ComponentSpawnInfos.push_back(info);
            // �������ʣ��������ݹ鴦��
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

        // Destroy����������ָ��ʵ��
        void destroyEntity(Entity entity)
        {
            // ����ʵ��
            if (auto it = world_.entities_.find(entity); it != world_.entities_.end())
            {
                // ����ʵ����������
                for (auto& [index, component] : it->second)
                {
                    auto& componentInfo = world_.componentMap_[index];  // ��ȡ�����Ϣ
                    componentInfo.pool.Destroy(component);              // �������ʵ��
                    componentInfo.RemoveEntity(entity);                 // ��ʵ�弯�����Ƴ�
                }
                world_.entities_.erase(it);  // ��ʵ��ӳ�����ɾ��ʵ��
            }
        }

        void setResource(const ResourceSpawnInfo& Info)
        {
            auto index = Info.index_;
            if (auto it = world_.resources_.find(index); it != world_.resources_.end())
            {
                // ���ͷ�ԭ����Դ
                if (it->second.resource) {
                    it->second.destroy(it->second.resource);
                }
                // ��������Դ
                it->second.resource = Info.create_();
            }
            else
            {
                auto NewIt = world_.resources_.emplace(index, World::ResourceInfo(
                    Info.destroy_
                ));

                // ʹ�ô����resource��������Դ��������Ϊһ����
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
        World& world_;  // World����

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