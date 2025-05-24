#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// class UnionFind {
// public:
//     explicit UnionFind(int size)
//     {
//         resize(size);
//     }
//
//     int find(int x) const noexcept
//     {
//         while (x != parent[x])
//             x = parent[x];
//         return x;
//     }
//
//     int find(int x) noexcept
//     {
//         int p = std::as_const(*this).find(x);
//         while (x != parent[x])
//             x = std::exchange(parent[x], p);
//         return x;
//     }
//
//     void join(int left, int right)
//     {
//         left = find(left);
//         right = find(right);
//
//         if (rank[left] >= rank[right]) {
//             parent[right] = left;
//             if (rank[left] == rank[right])
//                 ++rank[left];
//         } else {
//             parent[left] = right;
//         }
//     }
//
//     inline void disjoin(int x) noexcept
//     {
//         parent[x] = x;
//     }
//
//     inline bool connected(int left, int right) const noexcept
//     {
//         return find(left) == find(right);
//     }
//
//     inline int get_size() const noexcept
//     {
//         return parent.size();
//     }
//
//     void resize(int size)
//     {
//         int prev_size = get_size();
//         parent.resize(size);
//         rank.resize(size);
//
//         for (int i = prev_size; i < size; ++i) {
//             parent[i] = i;
//             rank[i] = i;
//         }
//     }
//
//     void reset(int size = -1)
//     {
//         if (size < 0)
//             size = get_size();
//         resize(0);
//         resize(size);
//     }
//
// private:
//     std::vector<int> parent, rank;
// };

template<typename T>
struct TreeNode {
    TreeNode *left = nullptr, *right = nullptr;
    T val;

    explicit TreeNode(T&& val = T(), TreeNode *left = nullptr, TreeNode *right = nullptr)
        : val(std::move(val)), left(left), right(right)
    {
    }

    explicit TreeNode(TreeNode *left, TreeNode *right) : left(left), right(right)
    {
    }
};

template<typename T, typename F = void (*)(TreeNode<T> *)>
class BTsGeneratorWithNLeaves {
    using Node = TreeNode<T>;

public:
    explicit BTsGeneratorWithNLeaves(F on_tree_create)
        : on_tree_create(on_tree_create)
    {
    }

    void generate(int nr_leaves)
    {
        if (nr_leaves <= 0)
            return;
        reset(nr_leaves);
        gen();
    }

private:
    void gen()
    {
        int prev_i = 0;
        for (int i = 1; i < leaves.size(); ++i) {
            if (leaves[i] == nullptr)
                continue;

            leaves[prev_i] = use_node(leaves[prev_i], leaves[i]);
            leaves[i] = nullptr;

            gen();

            leaves[i] = leaves[prev_i]->right;
            leaves[prev_i] = leaves[prev_i]->left;
            unuse_last_used_node();
            prev_i = i;
        }
        if (0 == prev_i)
            on_tree_create(leaves[0]);
    }

    void reset(int nr_leaves)
    {
        nodeset.resize(nr_leaves * 2 - 1);
        leaves.resize(nr_leaves);
        for (int i = 0; i < nr_leaves; ++i)
            leaves[i] = &nodeset[i];
        used_nodes = leaves.size();
    }

    Node *use_node(Node *left, Node *right)
    {
        Node *node = &nodeset[used_nodes++];
        node->left = left;
        node->right = right;
        return node;
    }

    void unuse_last_used_node()
    {
        --used_nodes;
    }

    std::vector<Node> nodeset;
    std::vector<Node *> leaves;
    F on_tree_create;
    int used_nodes = 0;
};


template<typename T>
void traverse_inorder(TreeNode<T> *root, std::ostream& os, int level = 0)
{
    if (! root)
        return;

    traverse_inorder(root->left, os, level + 1);
    for (int i = 0; i < level; ++i)
        os << '\t';
    os << root->val << std::endl;
    traverse_inorder(root->right, os, level + 1);
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
        return EXIT_FAILURE;

    int nr_leaves = atoi(argv[1]);
    auto tree_traverse = [](TreeNode<int> *root)
    {
        static int i = 0;
        std::cout << "Binary Tree #" << ++i << std::endl;
        traverse_inorder(root, std::cout);
    };

    BTsGeneratorWithNLeaves<int>{tree_traverse}.generate(nr_leaves);
    return 0;
}
