#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <algorithm>

// ������������͵�SparseSetʵ�֣�ʹ�ù̶���С��ҳ�ṹ
template <typename T, size_t PageSize,
    typename = std::enable_if_t<std::is_integral_v<T>>>
class SparseSet final
{
private:
    std::vector<T> dense;                  // �洢ʵ�����ݵ��ܼ�����
    std::vector<std::unique_ptr<std::array<T, PageSize>>> sparse;  // �洢������ϡ��ҳ
    static constexpr T null = std::numeric_limits<T>::max();       // ��ʾ��ֵ���ڱ�

    // ����Ԫ�����ڵ�ҳ����
    size_t getPageIndex(T element) const noexcept {
        return static_cast<size_t>(element) / PageSize;
    }

    // ����Ԫ����ҳ�ڵ�λ��
    size_t getPositionInPage(T element) const noexcept {
        return static_cast<size_t>(element) % PageSize;
    }

    // ȷ��ϡ��ҳ���ڣ�����������򴴽�
    void ensurePageExists(size_t pageIndex) {
        if (pageIndex >= sparse.size()) {
            sparse.resize(pageIndex + 1);
        }
        if (!sparse[pageIndex]) {
            sparse[pageIndex] = std::make_unique<std::array<T, PageSize>>();
            // ��ʼ����ҳ������λ�ö���Ϊnull
            std::fill(sparse[pageIndex]->begin(), sparse[pageIndex]->end(), null);
        }
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
        // ���Ԫ���Ƿ�Ϊnull����Чֵ��
        if (element == null) {
            throw std::invalid_argument("Cannot insert null value");
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);

        // ȷ��ҳ����
        ensurePageExists(pageIndex);

        // ���Ԫ���Ƿ��Ѵ���
        if (sparse[pageIndex]->at(pos) != null) {
            return false; // Ԫ���Ѵ���
        }

        // ��dense���������Ԫ��
        sparse[pageIndex]->at(pos) = static_cast<T>(dense.size());
        dense.push_back(element);
        return true;
    }

    // �Ƴ�Ԫ��
    bool erase(T element) {
        if (element == null || !contains(element)) {
            return false;
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);

        // ��ȡԪ����dense�����е�����
        const T denseIndex = sparse[pageIndex]->at(pos);

        // ��dense�����е����һ��Ԫ���ƶ�����ɾ��Ԫ�ص�λ��
        if (denseIndex != static_cast<T>(dense.size() - 1)) {
            const T lastElement = dense.back();
            dense[denseIndex] = lastElement;

            // �������һ��Ԫ����sparse�е�����
            const size_t lastPageIndex = getPageIndex(lastElement);
            const size_t lastPos = getPositionInPage(lastElement);
            sparse[lastPageIndex]->at(lastPos) = denseIndex;
        }

        // �Ƴ�dense�����е����һ��Ԫ��
        dense.pop_back();

        // ���sparse�е�λ��Ϊ��
        sparse[pageIndex]->at(pos) = null;

        return true;
    }

    // ���Ԫ���Ƿ����
    bool contains(T element) const {
        if (element == null) {
            return false;
        }

        const size_t pageIndex = getPageIndex(element);
        // ���ҳ�Ƿ����
        if (pageIndex >= sparse.size() || !sparse[pageIndex]) {
            return false;
        }

        const size_t pos = getPositionInPage(element);
        return sparse[pageIndex]->at(pos) != null;
    }

    // ��ȡԪ����dense�����е�����
    T indexOf(T element) const {
        if (!contains(element)) {
            return null;
        }

        const size_t pageIndex = getPageIndex(element);
        const size_t pos = getPositionInPage(element);
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

    // ��ռ���
    void clear() noexcept {
        dense.clear();
        sparse.clear();
    }

    // ��ȡʹ�õ�ҳ����
    size_t pageCount() const noexcept {
        return sparse.size();
    }

    // ��ȡҳ��С��ģ�������
    static constexpr size_t pageSize() noexcept {
        return PageSize;
    }
};
