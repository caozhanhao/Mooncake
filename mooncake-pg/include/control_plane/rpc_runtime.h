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
//
// Used by both CoordinatorHost and AgentHost.  Port 0 = auto-assign.
// Services are registered before start(), then start() launches the server.
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
// RpcClient — unified outbound RPC client (coroutine-based).
//
//   call<&Service::method>(addr, req)            — sync, blocks caller
//   callAsync<&Service::method>(addr, req, cb)   — async, callback on
//                                                    global executor thread
//   send<&Service::method>(addr, req)            — fire-and-forget
//
// Design:
//   - No dedicated background thread.  Async work is spawned as coroutines
//     on coro_io::get_global_executor().  This eliminates Head-of-Line
//     blocking: a slow / dead peer suspends only its own coroutine; other
//     peers make progress independently.
//   - Connection caching per address with shared_ptr<coro_rpc_client> for
//     safe concurrent access.  Even if tryReconnect() evicts an entry,
//     in-flight RPCs holding a shared_ptr copy keep the old client alive
//     until they complete — no use-after-free.
//   - send_request (pipeline / multiplexing) is used under the hood so
//     multiple concurrent requests to the same address share one TCP
//     connection.
//   - Sync call() creates a temporary coro_rpc_client each time to avoid
//     contention with the async path.  Control-plane sync RPC frequency is
//     low enough that the per-call connect cost is negligible.
// =========================================================================

class RpcClient {
   public:
    RpcClient() = default;
    ~RpcClient() = default;

    // ---- Sync RPC ----
    // Blocks the calling thread.  Uses call_for with an explicit timeout to
    // avoid blocking indefinitely on a dead peer.
    template <auto Func, typename Req>
    auto call(const std::string& addr, Req req,
              std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        using RpcResult = decltype(async_simple::coro::syncAwait(
            std::declval<coro_rpc::coro_rpc_client&>().call<Func>(req)));
        using ResponseType = decltype(std::declval<RpcResult>().value());

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
    // Spawns a coroutine on the global I/O executor.  The callback is
    // invoked on the global executor thread when the RPC completes (or
    // fails).  Callers should re-post to their own executor if needed
    // (existing callbacks already do this via executor_.post).
    template <auto Func, typename Req, typename Callback>
    void callAsync(const std::string& addr, Req req, Callback cb) {
        // Derive ResponseType from call<Func> (rpc_result<T>) —
        // the callback receives T directly, same as the sync path.
        using RpcResult = decltype(async_simple::coro::syncAwait(
            std::declval<coro_rpc::coro_rpc_client&>().call<Func>(req)));
        using ResponseType = decltype(std::declval<RpcResult>().value());

        auto state = state_;
        auto task =
            [state, addr, req = std::move(req),
             cb = std::move(cb)]() mutable -> async_simple::coro::Lazy<void> {
            auto client = co_await getOrCreateClientAsync(state, addr);
            if (!client) {
                cb(ResponseType{});
                co_return;
            }

            try {
                // call<Func> returns Lazy<rpc_result<T>> — single co_await.
                auto result = co_await client->call<Func>(req);
                if (result) {
                    cb(std::move(result.value()));
                } else {
                    LOG(ERROR) << "RpcClient: async rpc to " << addr
                               << " failed: " << result.error().msg;
                    cb(ResponseType{});
                }
            } catch (const std::exception& e) {
                LOG(ERROR) << "RpcClient: async rpc caught exception: "
                           << e.what();
                cb(ResponseType{});
            }
        }();

        auto executor = coro_io::get_global_executor();
        std::move(task).via(executor).start([](auto&&) {});
    }

    // ---- Fire-and-forget ----
    template <auto Func, typename Req>
    void send(const std::string& addr, Req req) {
        auto state = state_;
        auto task =
            [state, addr,
             req = std::move(req)]() mutable -> async_simple::coro::Lazy<void> {
            auto client = co_await getOrCreateClientAsync(state, addr);
            if (!client) co_return;
            try {
                // call<Func> returns Lazy<rpc_result<T>> — single co_await.
                // We discard the result for fire-and-forget.
                co_await client->call<Func>(req);
            } catch (...) {
            }
        }();

        auto executor = coro_io::get_global_executor();
        std::move(task).via(executor).start([](auto&&) {});
    }

    // ---- Connection health ----
    bool isConnected(const std::string& addr) const;
    bool tryReconnect(const std::string& addr);

   private:
    // Shared state that outlives any particular RpcClient instance.
    // In-flight coroutines hold a shared_ptr copy, so the client map
    // and mutex remain valid even after RpcClient is destroyed.
    struct SharedState {
        std::mutex mutex;
        std::unordered_map<std::string,
                           std::shared_ptr<coro_rpc::coro_rpc_client>>
            clients;
    };
    std::shared_ptr<SharedState> state_ = std::make_shared<SharedState>();

    // Coroutine-based connect + cache lookup.  Does NOT block an OS thread
    // — co_await suspends the coroutine, letting other work progress.
    static async_simple::coro::Lazy<std::shared_ptr<coro_rpc::coro_rpc_client>>
    getOrCreateClientAsync(std::shared_ptr<SharedState> state,
                           const std::string& addr);
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_RPC_RUNTIME_H
