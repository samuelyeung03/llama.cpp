#include "h3-quic.h"
#include "transport.h"
#include "ggml-impl.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

extern "C" void ggml_log_internal(enum ggml_log_level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

namespace {
struct entry_t {
    socket_ptr socket;
    bool listener;
    std::mutex operation;
    std::atomic<bool> closing{false};
    unsigned active = 0;
    entry_t(socket_ptr value, bool listening) : socket(std::move(value)), listener(listening) {}
};

std::mutex registry_mutex;
std::unordered_map<h3q_handle, std::shared_ptr<entry_t>> registry;
h3q_handle next_handle = 1;
unsigned creating = 0;
bool configured = false;
std::array<std::string, 7> configuration;

struct lease_t {
    std::shared_ptr<entry_t> entry;
    explicit lease_t(h3q_handle handle) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto found = registry.find(handle);
        if (found != registry.end()) {
            entry = found->second;
            ++entry->active;
        }
    }
    ~lease_t() {
        if (entry) {
            std::lock_guard<std::mutex> lock(registry_mutex);
            --entry->active;
        }
    }
};

struct creation_t {
    bool admitted = false;
    creation_t() {
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (configured) {
            ++creating;
            admitted = true;
        }
    }
    ~creation_t() {
        if (admitted) {
            std::lock_guard<std::mutex> lock(registry_mutex);
            --creating;
        }
    }
};

template<class function_t> int protect(function_t function) noexcept {
    try {
        return function();
    } catch (...) {
        return H3Q_INTERNAL;
    }
}

bool text(const char * value, size_t maximum) {
    if (!value || !*value) return false;
    const size_t length = strnlen(value, maximum + 1);
    if (length > maximum) return false;
    for (size_t i = 0; i < length; ++i) {
        if (static_cast<unsigned char>(value[i]) < 33 || static_cast<unsigned char>(value[i]) > 126) return false;
    }
    return true;
}

bool credential(const char * path, bool secret) {
    if (!path || path[0] != '/' || strnlen(path, 4097) > 4096) return false;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat info = {};
    bool valid = fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 && info.st_size <= 4 * 1024 * 1024;
    if (secret) valid = valid && info.st_uid == geteuid() && (info.st_mode & 077) == 0 && info.st_nlink == 1;
    ::close(fd);
    return valid;
}

bool ipv4(const char * value) {
    in_addr address = {};
    return value && inet_pton(AF_INET, value, &address) == 1 && address.s_addr != INADDR_ANY;
}

int register_socket(socket_ptr socket, bool listener, h3q_handle * out) {
    auto entry = std::make_shared<entry_t>(std::move(socket), listener);
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (next_handle == 0 || next_handle == std::numeric_limits<h3q_handle>::max()) return H3Q_INTERNAL;
    const h3q_handle handle = next_handle++;
    registry.emplace(handle, std::move(entry));
    *out = handle;
    return H3Q_OK;
}

template<class function_t> int operation(h3q_handle handle, bool listener, function_t function) {
    lease_t lease(handle);
    if (!lease.entry) return H3Q_BAD_HANDLE;
    if (lease.entry->listener != listener) return H3Q_INVALID;
    if (lease.entry->closing) return H3Q_CLOSED;
    std::unique_lock<std::mutex> lock(lease.entry->operation, std::try_to_lock);
    if (!lock.owns_lock()) return H3Q_BUSY;
    if (lease.entry->closing) return H3Q_CLOSED;
    if (!function(lease.entry->socket)) return lease.entry->closing ? H3Q_CLOSED : H3Q_IO_ERROR;
    return H3Q_OK;
}
} // namespace

extern "C" H3Q_API uint32_t h3q_abi_version(void) { return H3Q_ABI_VERSION; }

extern "C" H3Q_API int h3q_configure(const char * cert, const char * key, const char * ca,
                                     const char * sni, const char * cc, const char * alpn,
                                     uint32_t timeout_ms) {
    return protect([&]() -> int {
        if (!credential(cert, false) || !credential(key, true) || !credential(ca, false) ||
            !text(sni, 253) || !text(alpn, 255) || !cc ||
            (std::strcmp(cc, "newreno") != 0 && std::strcmp(cc, "cubic") != 0 && std::strcmp(cc, "bbr") != 0) ||
            timeout_ms < 100 || timeout_ms > 300000) return H3Q_INVALID;
        const std::array<std::string, 7> requested = {cert, key, ca, sni, cc, alpn, std::to_string(timeout_ms)};
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (!registry.empty() || creating != 0) {
            return configured && requested == configuration ? H3Q_OK : H3Q_BUSY;
        }
        configured = false;
        const char * names[] = {
            "GGML_RPC_QUIC_CERT", "GGML_RPC_QUIC_KEY", "GGML_RPC_QUIC_CA",
            "GGML_RPC_QUIC_SNI", "GGML_RPC_QUIC_CC", "GGML_RPC_QUIC_ALPN", "GGML_RPC_QUIC_IO_TIMEOUT_MS"
        };
        for (size_t i = 0; i < requested.size(); ++i) {
            if (setenv(names[i], requested[i].c_str(), 1) != 0) return H3Q_INTERNAL;
        }
        configuration = requested;
        configured = true;
        return H3Q_OK;
    });
}

extern "C" H3Q_API int h3q_listen(const char * address, uint16_t port, h3q_handle * out) {
    return protect([&]() -> int {
        if (!out || !ipv4(address)) return H3Q_INVALID;
        *out = 0;
        creation_t creation;
        if (!creation.admitted) return H3Q_INVALID;
        auto socket = socket_t::create_server(address, port);
        if (!socket) return H3Q_IO_ERROR;
        return register_socket(std::move(socket), true, out);
    });
}

extern "C" H3Q_API int h3q_connect(const char * address, uint16_t port, h3q_handle * out) {
    return protect([&]() -> int {
        if (!out || !ipv4(address) || port == 0) return H3Q_INVALID;
        *out = 0;
        creation_t creation;
        if (!creation.admitted) return H3Q_INVALID;
        auto socket = socket_t::connect(address, port);
        if (!socket) return H3Q_IO_ERROR;
        return register_socket(std::move(socket), false, out);
    });
}

extern "C" H3Q_API int h3q_accept(h3q_handle handle, h3q_handle * out) {
    return protect([&]() -> int {
        if (!out) return H3Q_INVALID;
        *out = 0;
        lease_t lease(handle);
        if (!lease.entry) return H3Q_BAD_HANDLE;
        if (!lease.entry->listener) return H3Q_INVALID;
        if (lease.entry->closing) return H3Q_CLOSED;
        std::unique_lock<std::mutex> lock(lease.entry->operation, std::try_to_lock);
        if (!lock.owns_lock()) return H3Q_BUSY;
        auto socket = lease.entry->socket->accept();
        if (!socket) return lease.entry->closing ? H3Q_CLOSED : H3Q_IO_ERROR;
        if (lease.entry->closing) {
            socket->close();
            return H3Q_CLOSED;
        }
        return register_socket(std::move(socket), false, out);
    });
}

extern "C" H3Q_API int h3q_send_exact(h3q_handle handle, const void * data, uint64_t bytes) {
    return protect([&]() -> int {
        if ((!data && bytes) || bytes > H3Q_MAX_CALL_BYTES) return H3Q_INVALID;
        return operation(handle, false, [&](const socket_ptr & socket) { return socket->send_data(data, static_cast<size_t>(bytes)); });
    });
}

extern "C" H3Q_API int h3q_recv_exact(h3q_handle handle, void * data, uint64_t bytes) {
    return protect([&]() -> int {
        if ((!data && bytes) || bytes > H3Q_MAX_CALL_BYTES) return H3Q_INVALID;
        return operation(handle, false, [&](const socket_ptr & socket) { return socket->recv_data(data, static_cast<size_t>(bytes)); });
    });
}

extern "C" H3Q_API int h3q_flush(h3q_handle handle) {
    return protect([&]() { return operation(handle, false, [](const socket_ptr & socket) { return socket->flush(); }); });
}

extern "C" H3Q_API int h3q_drain_acked(h3q_handle handle) {
    return protect([&]() { return operation(handle, false, [](const socket_ptr & socket) { return socket->drain_acked(); }); });
}

extern "C" H3Q_API int h3q_shutdown(h3q_handle handle) {
    return protect([&]() -> int {
        lease_t lease(handle);
        if (!lease.entry) return H3Q_BAD_HANDLE;
        lease.entry->closing = true;
        lease.entry->socket->shutdown();
        return H3Q_OK;
    });
}

extern "C" H3Q_API int h3q_close(h3q_handle handle) {
    return protect([&]() -> int {
        lease_t lease(handle);
        if (!lease.entry) return H3Q_BAD_HANDLE;
        lease.entry->closing = true;
        lease.entry->socket->close();
        return H3Q_OK;
    });
}

extern "C" H3Q_API int h3q_free(h3q_handle handle) {
    return protect([&]() -> int {
        std::shared_ptr<entry_t> released;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            auto found = registry.find(handle);
            if (found == registry.end()) return H3Q_BAD_HANDLE;
            if (found->second->active != 0) return H3Q_BUSY;
            if (!found->second->closing) return H3Q_INVALID;
            released = found->second;
            registry.erase(found);
        }
        return H3Q_OK;
    });
}

extern "C" H3Q_API int h3q_get_stats(h3q_handle handle, h3q_stats * out, uint32_t struct_bytes) {
    return protect([&]() -> int {
        if (!out || struct_bytes != sizeof(h3q_stats)) return H3Q_INVALID;
        lease_t lease(handle);
        if (!lease.entry) return H3Q_BAD_HANDLE;
        rpc_quic_stats_t native;
        if (!lease.entry->socket->get_quic_stats(&native)) return H3Q_IO_ERROR;
        *out = {};
        out->abi_version = H3Q_ABI_VERSION;
        out->struct_bytes = sizeof(h3q_stats);
        out->udp_sent_engine = native.udp_sent_engine;
        out->udp_received_engine = native.udp_received_engine;
        out->app_sent = native.app_sent;
        out->app_received = native.app_received;
        out->udp_send_drops_engine = native.udp_send_drops_engine;
        out->udp_parse_errors_engine = native.udp_parse_errors_engine;
        out->socket_receive_buffer = native.socket_receive_buffer;
        out->socket_send_buffer = native.socket_send_buffer;
        out->local_port = native.local_port;
        out->peer_port = native.peer_port;
        out->failed = native.failed;
        std::memcpy(out->local_ipv4, native.local_ipv4, sizeof(out->local_ipv4));
        std::memcpy(out->peer_ipv4, native.peer_ipv4, sizeof(out->peer_ipv4));
        return H3Q_OK;
    });
}
