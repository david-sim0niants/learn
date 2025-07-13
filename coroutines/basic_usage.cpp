#include <coroutine>
#include <iostream>

template<typename T>
struct Generator {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type {
        T current_value;

        auto get_return_object()
        {
            return Generator{handle_type::from_promise(*this)};
        }

        auto initial_suspend()
        {
            return std::suspend_always{};
        }

        auto final_suspend() noexcept
        {
            return std::suspend_always{};
        }

        void unhandled_exception()
        {
            std::exit(1);
        }

        auto yield_value(T value)
        {
            current_value = value;
            return std::suspend_always{};
        }

        void return_void()
        {
        }
    };

    handle_type coro;

    ~Generator()
    {
        if (coro)
            coro.destroy();
    }

    bool next()
    {
        if (! coro.done()) {
            coro.resume();
        }
        return ! coro.done();
    }

    T value() const
    {
        return coro.promise().current_value;
    }
};

Generator<int> count(int max)
{
    for (int i = 0; i < max; ++i)
        co_yield i;
}

int main()
{
    auto gen = count(5);
    while (gen.next())
        std::cout << gen.value() << std::endl;
    return 0;
}
