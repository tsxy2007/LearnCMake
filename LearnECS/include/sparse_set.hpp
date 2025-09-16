#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <functional>

// ������������͵�SparseSetʵ�֣�ʹ�ù̶���С��ҳ�ṹ
template <typename T, size_t PageSize,
    typename = std::enable_if_t<std::is_integral_v<T>>>
class SparseSet final
{
private:
    std::vector<T> dense;                  // �洢ʵ�����ݵ��ܼ�����
    std::vector<std::unique_ptr<std::array<T, PageSize>>> sparse;  // �洢������ϡ��ҳ
    static constexpr T null = std::numeric_limits<T>::max();       // ��ʾ��ֵ���ڱ�

    // ����Ԫ�����ڵ�ҳ������ҳ��λ��
    std::pair<size_t, size_t> getPageAndPosition(T element) const noexcept {
        size_t index = static_cast<size_t>(element);
        return { index / PageSize, index % PageSize };
    }

    // ȷ��ϡ��ҳ���ڣ�����������򴴽�
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

    // ���ҳ�Ƿ������Ԫ��λ�÷ǿ�
    bool isElementValid(T element) const noexcept {
        if (element == null) return false;
        auto [pageIndex, pos] = getPageAndPosition(element);
        if (pageIndex >= sparse.size() || !sparse[pageIndex]) return false;
        return sparse[pageIndex]->at(pos) != null;
    }

public:
    // ���캯��
    SparseSet() = default;

    // ��ֹ��������
    SparseSet(const SparseSet&) = delete;
    SparseSet& operator=(const SparseSet&) = delete;

    // �����ƶ�����
    SparseSet(SparseSet&&) = default;
    SparseSet& operator=(SparseSet&&) = default;

    // ��������
    ~SparseSet() = default;

    // ����Ԫ��
    bool insert(T element) {
        if (element == null) {
            throw std::invalid_argument("Cannot insert null value");
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        ensurePageExists(pageIndex);

        if (sparse[pageIndex]->at(pos) != null) {
            return false; // Ԫ���Ѵ���
        }

        sparse[pageIndex]->at(pos) = static_cast<T>(dense.size());
        dense.push_back(element);
        return true;
    }

    // �Ƴ�Ԫ��
    bool erase(T element) {
        if (element == null || !isElementValid(element)) {
            return false;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        T denseIndex = sparse[pageIndex]->at(pos);

        // �����޷������
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

    // ���Ԫ���Ƿ����
    bool contains(T element) const {
        return isElementValid(element);
    }

    // ��ȡԪ����dense�����е�����
    T indexOf(T element) const {
        if (!isElementValid(element)) {
            return null;
        }

        auto [pageIndex, pos] = getPageAndPosition(element);
        return sparse[pageIndex]->at(pos);
    }

    // ��ȡdense������ָ��������Ԫ��
    T at(size_t index) const {
        if (index >= dense.size()) {
            throw std::out_of_range("Index out of range");
        }
        return dense[index];
    }

    // ��ȡԪ������
    size_t size() const noexcept {
        return dense.size();
    }

    // ����Ƿ�Ϊ��
    bool empty() const noexcept {
        return dense.empty();
    }

    // ��ռ��ϣ�����ҳ�ṹ��
    void clear() noexcept {
        dense.clear();
        for (auto& page : sparse) {
            if (page) {
                std::fill(page->begin(), page->end(), null);
            }
        }
    }

    // ��ȡʹ�õ�ҳ����
    size_t pageCount() const noexcept {
        return sparse.size();
    }

    // ��ȡҳ��С��ģ�������
    static constexpr size_t pageSize() noexcept {
        return PageSize;
    }

    // ���foreachѭ������
    template<typename Func>
    void foreach(Func&& func) const {
        for (const T& element : dense) {
            func(element);
        }
    }

    // ��Ӵ�������foreachѭ������
    template<typename Func>
    void foreach_with_index(Func&& func) const {
        for (size_t i = 0; i < dense.size(); ++i) {
            func(dense[i], i);
        }
    }

    // ������֧��
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