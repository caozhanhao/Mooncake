#ifndef MOONCAKE_PG_RPC_RUNTIME_H
#define MOONCAKE_PG_RPC_RUNTIME_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>

#include <glog/logging.h>

// Workaround for yalantinglibs #1152
#include <csignal>

#include <ylt/coro_rpc/coro_rpc_client.hpp>
#include <ylt/coro_rpc/coro_rpc_server.hpp>
#include <ylt/coro_io/coro_io.hpp>

#include <async_simple/coro/SyncAwait.h>
#include <async_simple/coro/Lazy.h>

namespace mooncake {

// =========================================================================
// RpcServer — thin wrapper around coro_rpc::coro_rpc_server.
// =========================================================================

class RpcServer {
   public:
    explicit RpcServer(uint16_t port = 0, unsigned thread_num = 2);

    // Register coro_rpc service handler(s).  coro_rpc expects non-type
    // template parameters (function pointers) and a raw pointer to the
    // service instance.  Caller must keep *impl alive for the server
    // lifetime.
    template <auto First, auto... Rest>
    void registerHandler(util::class_type_t<decltype(First)>* impl) {
        server_->register_handler<First, Rest...>(impl);
    }

    bool start();
    uint16_t getPort() const;
    std::string getListenAddr(const std::string& host_ip) const;
    void shutdown();

   private:
    std::unique_ptr<coro_rpc::coro_rpc_server> server_;
    uint16_t port_;
    unsigned thread_num_;
};

// =========================================================================
// RpcClient helpers — free function coroutines.
//
// GCC 11 fails to resolve non-type template parameters (pointer-to-virtual-
// member-function) forwarded through lambdas inside coroutine contexts.
// Extracting the coroutine into a free function template avoids this.
// =========================================================================

namespace rpc_detail {

// Shared state for connection caching (forward declaration).
struct RpcSharedState {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<coro_rpc::coro_rpc_client>>
        clients;
};

// Coroutine-based connect + cache lookup.
async_simple::coro::Lazy<std::shared_ptr<coro_rpc::coro_rpc_client>>
getOrCreateClientAsync(std::shared_ptr<RpcSharedState> state,
                       const std::string& addr);

// Fire-and-forget coroutine: connect, send_request, discard result.
// Uses send_request<Func, Req> with explicit Args... to avoid GCC 11
// overload resolution issues with non-type template parameters.
template <auto Func, typename Req>
async_simple::coro::Lazy<void> sendCoroutine(
    std::shared_ptr<RpcSharedState> state, const std::string& addr, Req req) {
    auto client = co_await getOrCreateClientAsync(state, addr);
    if (!client) co_return;
    try {
        auto send_lazy =
            co_await client->send_request<Func, Req>(std::move(req));
        co_await std::move(send_lazy);
    } catch (...) {
    }
}

// Async call with callback coroutine: connect, send_request, invoke callback.
// Uses send_request<Func, Req> with explicit Args... to avoid GCC 11
// overload resolution issues with non-type template parameters.
// Callback type is templated to avoid requiring std::function (which needs
// CopyConstructible) — the caller's lambda is moved into cb.
template <auto Func, typename Req, typename ResponseType, typename Callback>
async_simple::coro::Lazy<void> callAsyncCoroutine(
    std::shared_ptr<RpcSharedState> state, const std::string& addr, Req req,
    Callback cb) {
    auto client = co_await getOrCreateClientAsync(state, addr);
    if (!client) {
        cb(ResponseType{});
        co_return;
    }
    try {
        // send_request returns Lazy<Lazy<async_rpc_result<T>>>.
        // Outer co_await: request sent.  Inner co_await: response received.
        auto send_lazy =
            co_await client->send_request<Func, Req>(std::move(req));
        auto res = co_await std::move(send_lazy);
        if (res) {
            cb(std::move(res.value().result()));
        } else {
            LOG(ERROR) << "RpcClient: async rpc to " << addr
                       << " failed: " << res.error().msg;
            cb(ResponseType{});
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "RpcClient: async rpc caught exception: " << e.what();
        cb(ResponseType{});
    }
}

}  // namespace rpc_detail

// =========================================================================
// RpcClient — unified outbound RPC client.
//
//   call<&Service::method>(addr, req)            — sync, blocks caller
//   callAsync<&Service::method>(addr, req, cb)   — async, callback on
//                                                    global executor thread
//   send<&Service::method>(addr, req)            — fire-and-forget
// =========================================================================

class RpcClient {
   public:
    RpcClient() = default;
    ~RpcClient() = default;

    // ---- Sync RPC ----
    template <auto Func, typename Req>
    auto call(const std::string& addr, Req req,
              std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        // Derive ResponseType directly from the function pointer to avoid
        // instantiating call<Func> (which has overload resolution issues
        // with GCC 11 and non-type template parameters).
        using ResponseType = decltype(coro_rpc::get_return_type<Func>());

        coro_rpc::coro_rpc_client::config config;
        config.connect_timeout_duration = std::chrono::seconds(3);
        coro_rpc::coro_rpc_client client(coro_io::get_global_executor(),
                                         config);

        auto ec = async_simple::coro::syncAwait(client.connect(addr));
        if (ec) {
            LOG(ERROR) << "RpcClient: call connect failed to " << addr << ": "
                       << ec.message();
            return ResponseType{};
        }
        auto result =
            async_simple::coro::syncAwait(client.call_for<Func>(timeout, req));
        if (!result) {
            LOG(ERROR) << "RpcClient: call RPC failed: " << result.error().msg;
            return ResponseType{};
        }
        return std::move(result.value());
    }

    // ---- Async RPC with callback ----
    template <auto Func, typename Req, typename Callback>
    void callAsync(const std::string& addr, Req req, Callback cb) {
        // Derive ResponseType directly from the function pointer to avoid
        // instantiating call<Func> (which has overload resolution issues
        // with GCC 11 and non-type template parameters).
        using ResponseType = decltype(coro_rpc::get_return_type<Func>());

        auto task =
            rpc_detail::callAsyncCoroutine<Func, Req, ResponseType, Callback>(
                state_, addr, std::move(req), std::move(cb));

        auto executor = coro_io::get_global_executor();
        std::move(task).via(executor).start([](auto&&) {});
    }

    // ---- Fire-and-forget ----
    template <auto Func, typename Req>
    void send(const std::string& addr, Req req) {
        auto task =
            rpc_detail::sendCoroutine<Func, Req>(state_, addr, std::move(req));

        auto executor = coro_io::get_global_executor();
        std::move(task).via(executor).start([](auto&&) {});
    }

    // ---- Connection health ----
    bool isConnected(const std::string& addr) const;
    bool tryReconnect(const std::string& addr);

   private:
    std::shared_ptr<rpc_detail::RpcSharedState> state_ =
        std::make_shared<rpc_detail::RpcSharedState>();
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_RPC_RUNTIME_H
