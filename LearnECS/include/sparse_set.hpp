#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <algorithm>

// 仅针对整数类型的SparseSet实现，使用固定大小的页结构
template <typename T, size_t PageSize,
    typename = std::enable_if_t<std::is_integral_v<T>>>
class SparseSet final
{
private:
    std::vector<T> dense;                  // 存储实际数据的密集数组
    std::vector<std::unique_ptr<std::array<T, PageSize>>> sparse;  // 存储索引的稀疏页
    static constexpr T null = std::numeric_limits<T>::max();       // 表示空值的哨兵

    // 计算元素所在的页索引
    size_t getPageIndex(T element) const noexcept {
        return static_cast<size_t>(element) / PageSize;
    }

    // 计算元素在页内的位置
    size_t getPositionInPage(T element) const noexcept {
        return static_cast<size_t>(element) % PageSize;
    }

    // 确保稀疏页存在，如果不存在则创建
    void ensurePageExists(size_t pageIndex) {
        if (pageIndex >= sparse.size()) {
            sparse.resize(pageIndex + 1);
        }
        if (!sparse[pageIndex]) {
            sparse[pageIndex] = std::make_unique<std::array<T, PageSize>>();
            // 初始化新页，所有位置都设为null
            std::fill(sparse[pageIndex]->begin(), sparse[pageIndex]->end(), null);
        }
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
        // 检查元素是否为null（无效值）
        if (element == null) {
            throw std::invalid_argument("Cannot insert null value");
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);

        // 确保页存在
        ensurePageExists(pageIndex);

        // 检查元素是否已存在
        if (sparse[pageIndex]->at(pos) != null) {
            return false; // 元素已存在
        }

        // 在dense数组中添加元素
        sparse[pageIndex]->at(pos) = static_cast<T>(dense.size());
        dense.push_back(element);
        return true;
    }

    // 移除元素
    bool erase(T element) {
        if (element == null || !contains(element)) {
            return false;
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);

        // 获取元素在dense数组中的索引
        const T denseIndex = sparse[pageIndex]->at(pos);

        // 将dense数组中的最后一个元素移动到被删除元素的位置
        if (denseIndex != static_cast<T>(dense.size() - 1)) {
            const T lastElement = dense.back();
            dense[denseIndex] = lastElement;

            // 更新最后一个元素在sparse中的索引
            const size_t lastPageIndex = getPageIndex(lastElement);
            const size_t lastPos = getPositionInPage(lastElement);
            sparse[lastPageIndex]->at(lastPos) = denseIndex;
        }

        // 移除dense数组中的最后一个元素
        dense.pop_back();

        // 标记sparse中的位置为空
        sparse[pageIndex]->at(pos) = null;

        return true;
    }

    // 检查元素是否存在
    bool contains(T element) const {
        if (element == null) {
            return false;
        }

        const size_t pageIndex = getPageIndex(element);
        // 检查页是否存在
        if (pageIndex >= sparse.size() || !sparse[pageIndex]) {
            return false;
        }

        const size_t pos = getPositionInPage(element);
        return sparse[pageIndex]->at(pos) != null;
    }

    // 获取元素在dense数组中的索引
    T indexOf(T element) const {
        if (!contains(element)) {
            return null;
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);
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

    // 清空集合
    void clear() noexcept {
        dense.clear();
        sparse.clear();
    }

    // 获取使用的页数量
    size_t pageCount() const noexcept {
        return sparse.size();
    }

    // 获取页大小（模板参数）
    static constexpr size_t pageSize() noexcept {
        return PageSize;
    }
};
