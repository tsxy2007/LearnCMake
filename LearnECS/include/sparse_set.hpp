#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <functional>

// 用于实体组件类型的 SparseSet（稀疏集合）实现，使用固定大小的页结构来高效管理稀疏索引
template <typename T, size_t PageSize,
    typename = std::enable_if_t<std::is_integral_v<T>>>
class SparseSet final
{
private:
    std::vector<T> dense;                                              // 存储实际数据的密集数组，元素连续存放
    std::vector<std::unique_ptr<std::array<T, PageSize>>> sparse;      // 存储索引映射的稀疏页，每页 PageSize 个槽位
    static constexpr T null = std::numeric_limits<T>::max();           // 表示空值/无效值的标记

    // 计算元素所在的页索引和页内偏移位置
    std::pair<size_t, size_t> getPageAndPosition(T element) const noexcept {
        size_t index = static_cast<size_t>(element);
        return { index / PageSize, index % PageSize };
    }

    // 确保指定索引的稀疏页存在，如果不存在则分配新页并对所有槽位填充 null
    void ensurePageExists(size_t pageIndex) {
        if (pageIndex >= sparse.size()) {
            sparse.resize(pageIndex + 1);
        }
        if (!sparse[pageIndex]) {
            auto page = std::make_unique<std::array<T, PageSize>>();
            std::fill(page->begin(), page->end(), null);
            sparse[pageIndex] = std::move(page);
        }
    }

    // 检查元素是否有效——即该元素对应的稀疏页槽位是否非空
    bool isElementValid(T element) const noexcept {
        if (element == null) return false;
        auto [pageIndex, pos] = getPageAndPosition(element);
        if (pageIndex >= sparse.size() || !sparse[pageIndex]) return false;
        return sparse[pageIndex]->at(pos) != null;
    }

public:
    // 默认构造函数
    SparseSet() = default;

    // 禁止拷贝构造和拷贝赋值（稀疏集合不支持拷贝语义）
    SparseSet(const SparseSet&) = delete;
    SparseSet& operator=(const SparseSet&) = delete;

    // 允许移动构造和移动赋值
    SparseSet(SparseSet&&) = default;
    SparseSet& operator=(SparseSet&&) = default;

    // 析构函数
    ~SparseSet() = default;

    // 插入元素：将元素添加到 dense 数组末尾，并在 sparse 页中记录其索引
    // 返回 true 表示插入成功，返回 false 表示元素已存在
    bool insert(T element) {
        if (element == null) {
            throw std::invalid_argument("Cannot insert null value");
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        ensurePageExists(pageIndex);

        if (sparse[pageIndex]->at(pos) != null) {
            return false; // 元素已存在，不重复插入
        }

        sparse[pageIndex]->at(pos) = static_cast<T>(dense.size());
        dense.push_back(element);
        return true;
    }

    // 移除元素：使用 swap-and-pop 技巧，将 dense 末尾元素移到被删位置以保持数组紧凑
    // 返回 true 表示删除成功，返回 false 表示元素不存在
    bool erase(T element) {
        if (element == null || !isElementValid(element)) {
            return false;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        T denseIndex = sparse[pageIndex]->at(pos);

        // 边界条件检查：dense 为空时无法删除
        if (dense.empty()) return false;

        // 如果被删元素不是 dense 末尾，则将末尾元素移到空缺位置，并更新其 sparse 映射
        if (denseIndex != static_cast<T>(dense.size() - 1)) {
            T lastElement = dense.back();
            dense[denseIndex] = lastElement;

            auto [lastPageIndex, lastPos] = getPageAndPosition(lastElement);
            sparse[lastPageIndex]->at(lastPos) = denseIndex;
        }

        dense.pop_back();
        sparse[pageIndex]->at(pos) = null;
        return true;
    }

    // 检查元素是否存在于集合中
    bool contains(T element) const {
        return isElementValid(element);
    }

    // 获取元素在 dense 数组中的索引位置
    // 如果元素无效则返回 null
    T indexOf(T element) const {
        if (!isElementValid(element)) {
            return null;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        return sparse[pageIndex]->at(pos);
    }

    // 根据 dense 数组索引获取对应的元素值
    // 索引越界时抛出 std::out_of_range 异常
    T at(size_t index) const {
        if (index >= dense.size()) {
            throw std::out_of_range("Index out of range");
        }
        return dense[index];
    }

    // 获取集合中元素的个数
    size_t size() const noexcept {
        return dense.size();
    }

    // 判断集合是否为空
    bool empty() const noexcept {
        return dense.empty();
    }

    // 清空集合中的所有元素，但保留已分配的稀疏页结构以便复用
    void clear() noexcept {
        dense.clear();
        for (auto& page : sparse) {
            if (page) {
                std::fill(page->begin(), page->end(), null);
            }
        }
    }

    // 获取当前已分配的稀疏页数量
    size_t pageCount() const noexcept {
        return sparse.size();
    }

    // 获取页大小（模板参数 PageSize 的值）
    static constexpr size_t pageSize() noexcept {
        return PageSize;
    }

    // 遍历集合中的所有元素，对每个元素调用指定函数
    template<typename Func>
    void foreach(Func&& func) const {
        for (const T& element : dense) {
            func(element);
        }
    }

    // 带索引的遍历：对每个元素调用 func(元素值, 索引位置)
    template<typename Func>
    void foreach_with_index(Func&& func) const {
        for (size_t i = 0; i < dense.size(); ++i) {
            func(dense[i], i);
        }
    }

    // —— 迭代器支持，可以使用范围 for 循环遍历 ——
    typename std::vector<T>::const_iterator begin() const noexcept {
        return dense.begin();
    }

    typename std::vector<T>::const_iterator end() const noexcept {
        return dense.end();
    }

    typename std::vector<T>::const_iterator cbegin() const noexcept {
        return dense.cbegin();
    }

    typename std::vector<T>::const_iterator cend() const noexcept {
        return dense.cend();
    }
};