#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <functional>

// 仅针对整数类型的SparseSet实现，使用固定大小的页结构
template <typename T, size_t PageSize,
    typename = std::enable_if_t<std::is_integral_v<T>>>
class SparseSet final
{
private:
    std::vector<T> dense;                  // 存储实际数据的密集数组
    std::vector<std::unique_ptr<std::array<T, PageSize>>> sparse;  // 存储索引的稀疏页
    static constexpr T null = std::numeric_limits<T>::max();       // 表示空值的哨兵

    // 计算元素所在的页索引和页内位置
    std::pair<size_t, size_t> getPageAndPosition(T element) const noexcept {
        size_t index = static_cast<size_t>(element);
        return { index / PageSize, index % PageSize };
    }

    // 确保稀疏页存在，如果不存在则创建
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

    // 检查页是否存在且元素位置非空
    bool isElementValid(T element) const noexcept {
        if (element == null) return false;
        auto [pageIndex, pos] = getPageAndPosition(element);
        if (pageIndex >= sparse.size() || !sparse[pageIndex]) return false;
        return sparse[pageIndex]->at(pos) != null;
    }

public:
    // 构造函数
    SparseSet() = default;

    // 禁止拷贝操作
    SparseSet(const SparseSet&) = delete;
    SparseSet& operator=(const SparseSet&) = delete;

    // 允许移动操作
    SparseSet(SparseSet&&) = default;
    SparseSet& operator=(SparseSet&&) = default;

    // 析构函数
    ~SparseSet() = default;

    // 插入元素
    bool insert(T element) {
        if (element == null) {
            throw std::invalid_argument("Cannot insert null value");
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        ensurePageExists(pageIndex);

        if (sparse[pageIndex]->at(pos) != null) {
            return false; // 元素已存在
        }

        sparse[pageIndex]->at(pos) = static_cast<T>(dense.size());
        dense.push_back(element);
        return true;
    }

    // 移除元素
    bool erase(T element) {
        if (element == null || !isElementValid(element)) {
            return false;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        T denseIndex = sparse[pageIndex]->at(pos);

        // 避免无符号溢出
        if (dense.empty()) return false;

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

    // 检查元素是否存在
    bool contains(T element) const {
        return isElementValid(element);
    }

    // 获取元素在dense数组中的索引
    T indexOf(T element) const {
        if (!isElementValid(element)) {
            return null;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        return sparse[pageIndex]->at(pos);
    }

    // 获取dense数组中指定索引的元素
    T at(size_t index) const {
        if (index >= dense.size()) {
            throw std::out_of_range("Index out of range");
        }
        return dense[index];
    }

    // 获取元素数量
    size_t size() const noexcept {
        return dense.size();
    }

    // 检查是否为空
    bool empty() const noexcept {
        return dense.empty();
    }

    // 清空集合（保留页结构）
    void clear() noexcept {
        dense.clear();
        for (auto& page : sparse) {
            if (page) {
                std::fill(page->begin(), page->end(), null);
            }
        }
    }

    // 获取使用的页数量
    size_t pageCount() const noexcept {
        return sparse.size();
    }

    // 获取页大小（模板参数）
    static constexpr size_t pageSize() noexcept {
        return PageSize;
    }

    // 添加foreach循环功能
    template<typename Func>
    void foreach(Func&& func) const {
        for (const T& element : dense) {
            func(element);
        }
    }

    // 添加带索引的foreach循环功能
    template<typename Func>
    void foreach_with_index(Func&& func) const {
        for (size_t i = 0; i < dense.size(); ++i) {
            func(dense[i], i);
        }
    }

    // 迭代器支持
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