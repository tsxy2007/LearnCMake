#pragma once  // ��ֹͷ�ļ�����ΰ���

#include <vector>        // ����vector����
#include <algorithm>     // �����㷨��
#include <cassert>       // �������Ժ�
#include <unordered_map> // ������ϣ������
#include "sparse_set.hpp" // ����ϡ�輯��ͷ�ļ�

#define assertm(msg,expr) assert(((void)msg,(expr))) // �������Ϣ�Ķ��Ժ�

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
        using ComponentContainer = std::unordered_map<ComponentID, void*>;  // ����������ͱ���

        World() = default;
        World(World&&) = delete;
        World& operator=(World&&) = delete;
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
    };

    // Commonds�࣬����ִ�д���/����ʵ�������
    class Commonds final
    {
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
            Entity entity = EntityGenerator::Gen();  // ������ʵ��ID
            doSpawn(entity, std::forward<ComponentTypes>(components)...);  // ִ�д���
            return *this;  // ����������֧����ʽ����
        }

        // Destroy����������ָ��ʵ��
        Commonds Destroy(Entity entity)
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
            return *this;  // ����������֧����ʽ����
        }


        template<typename T>
        Commonds& SetResource(T&& resource)
        {
            auto index = IndexGetter<Resource>::Get<T>();
            if (auto it = world_.resources_.find(index); it != world_.resources_.end())
            {
                // ���ͷ�ԭ����Դ
                if (it->second.resource) {
                    it->second.destroy(it->second.resource);
                }
                // ��������Դ
                it->second.resource = new T(std::forward<T>(resource));
            }
            else
            {
                auto NewIt = world_.resources_.emplace(index, World::ResourceInfo(
                    [](void* instance) {delete (T*)instance; }
                ));

                // ʹ�ô����resource��������Դ��������Ϊһ����
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
        // doSpawn�������ݹ��Ϊʵ��������
        template<typename T, typename ... Remains>
        void doSpawn(Entity entity, T&& component, Remains&& ... remains)
        {
            auto Index = IndexGetter<Component>::Get<T>();  // ��ȡ�������ID
            // ������������δע��
            if (auto it = world_.componentMap_.find(Index); it == world_.componentMap_.end())
            {
                // ע�����������
                world_.componentMap_.emplace(Index, World::ComponentInfo(
                    []()->void* {return new T(); },           // ��������
                    [](void* instance) {delete (T*)instance; } // ���ٺ���
                ));
            }
            auto& componentInfo = world_.componentMap_[Index];  // ��ȡ�����Ϣ
            void* elem = componentInfo.pool.Create();           // �������ʵ��
            *((T*)elem) = std::forward<T>(component);           // ��ֵ�������
            componentInfo.AddEntity(entity);                    // ���ʵ�嵽����

            // ��ʵ��ӳ�����������
            auto it = world_.entities_.emplace(entity, World::ComponentContainer{});
            it.first->second[Index] = elem;

            // �������ʣ��������ݹ鴦��
            if constexpr (sizeof...(Remains) != 0)
            {
                doSpawn<Remains...>(entity, std::forward<Remains>(remains)...);
            }
        }
    private:
        World& world_;  // World����
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