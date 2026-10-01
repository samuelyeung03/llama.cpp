// Direct, authenticated QUIC stream transport for the ggml RPC socket API.
// A socket_t is used synchronously by one RPC dispatcher. The server processes
// accepted connections serially; only close() may run from another thread.
#include "transport.h"
#include "ggml-impl.h"
#include "picoquic.h"
#include "picoquic_qlog.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
constexpr size_t RING_BYTES = 8 * 1024 * 1024;
constexpr uint64_t FLOW_BYTES = 4 * 1024 * 1024;
constexpr uint64_t CWIN_BYTES = 8 * 1024 * 1024;
constexpr int SOCKET_BUFFER_REQUEST = 4 * 1024 * 1024;
constexpr int IO_TIMEOUT_MS = 300000;
constexpr uint64_t GRACEFUL_CLOSE_US = 10000000;
constexpr char ALPN[] = "hoshino-rpc/1";
constexpr char DEFAULT_SNI[] = "hoshino-rpc";

uint64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * required_env(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] != '/' || access(value, R_OK) != 0) {
        GGML_LOG_ERROR("QUIC requires readable absolute %s\n", name);
        return nullptr;
    }
    return value;
}

struct ring_t {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(RING_BYTES);
    size_t head = 0;
    size_t used = 0;
    size_t write(const uint8_t * src, size_t size) {
        const size_t n = std::min(size, bytes.size() - used);
        const size_t tail = (head + used) % bytes.size();
        const size_t first = std::min(n, bytes.size() - tail);
        memcpy(bytes.data() + tail, src, first);
        memcpy(bytes.data(), src + first, n - first);
        used += n;
        return n;
    }
    size_t read(uint8_t * dst, size_t size) {
        const size_t n = std::min(size, used);
        const size_t first = std::min(n, bytes.size() - head);
        memcpy(dst, bytes.data() + head, first);
        memcpy(dst + first, bytes.data(), n - first);
        head = (head + n) % bytes.size();
        used -= n;
        return n;
    }
};

struct channel_t {
    picoquic_cnx_t * cnx = nullptr;
    unsigned id = 0;
    ring_t tx;
    ring_t rx;
    bool ready = false;
    bool failed = false;
    bool accepted = false;
    bool flow_manual = false;
    bool flow_update = false;
    bool peer_fin = false;
    bool fin_requested = false;
    bool fin_released = false;
    bool close_sent = false;
    bool acked_empty = false;
    bool final_stats_written = false;
    uint64_t close_started_us = 0;
    std::atomic<bool> closing{false};
    uint64_t app_sent = 0;
    uint64_t app_received = 0;
};

struct engine_t : std::enable_shared_from_this<engine_t> {
    int fd = -1;
    int wake_fd = -1;
    picoquic_quic_t * quic = nullptr;
    sockaddr_storage local = {};
    std::atomic<bool> stopping{false};
    std::mutex mu;
    std::condition_variable cv;
    std::thread io_thread;
    bool server = false;
    int timeout_ms = IO_TIMEOUT_MS;
    int idle_timeout_ms = 600000;
    std::string cc;
    std::string stats_dir;
    std::string qlog_dir;
    uint64_t wire_sent = 0;
    uint64_t wire_received = 0;
    uint64_t udp_send_drops = 0;
    uint64_t udp_parse_errors = 0;
    int socket_receive_buffer = 0;
    int socket_send_buffer = 0;
    bool last_progress = false;
    uint64_t last_stats_us = 0;
    unsigned next_id = 0;
    unsigned engine_id = []() { static std::atomic<unsigned> seq{0}; return seq++; }();
    std::unordered_map<picoquic_cnx_t *, std::shared_ptr<channel_t>> channels;
    std::deque<std::shared_ptr<channel_t>> pending;

    ~engine_t() {
        stopping = true;
        wake();
        cv.notify_all();
        if (io_thread.joinable()) io_thread.join();
        if (quic) picoquic_free(quic);
        if (fd >= 0) ::close(fd);
        if (wake_fd >= 0) ::close(wake_fd);
    }
    void wake() {
        if (wake_fd >= 0) { uint64_t one = 1; (void)::write(wake_fd, &one, sizeof(one)); }
    }
    bool configure_socket_buffers() {
        if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &SOCKET_BUFFER_REQUEST, sizeof(SOCKET_BUFFER_REQUEST)) != 0 ||
            setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &SOCKET_BUFFER_REQUEST, sizeof(SOCKET_BUFFER_REQUEST)) != 0) {
            GGML_LOG_ERROR("QUIC socket buffer request failed: %s\n", strerror(errno));
            return false;
        }
        socklen_t len = sizeof(int);
        if (getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &socket_receive_buffer, &len) != 0) return false;
        len = sizeof(int);
        if (getsockopt(fd, SOL_SOCKET, SO_SNDBUF, &socket_send_buffer, &len) != 0) return false;
        GGML_LOG_INFO("QUIC UDP socket buffers requested=%d effective_receive=%d effective_send=%d\n",
                      SOCKET_BUFFER_REQUEST, socket_receive_buffer, socket_send_buffer);
        return true;
    }
    bool init(bool is_server) {
        server = is_server;
        const char * cert = required_env("GGML_RPC_QUIC_CERT");
        const char * key = required_env("GGML_RPC_QUIC_KEY");
        const char * ca = required_env("GGML_RPC_QUIC_CA");
        if (!cert || !key || !ca) return false;
        const char * cc_env = std::getenv("GGML_RPC_QUIC_CC");
        cc = cc_env ? cc_env : "cubic";
        if (cc != "newreno" && cc != "cubic" && cc != "bbr") {
            GGML_LOG_ERROR("QUIC CC must be newreno, cubic, or bbr\n");
            return false;
        }
        const char * dir = std::getenv("GGML_RPC_QUIC_STATS_DIR");
        if (dir) stats_dir = dir;
        const char * qlog = std::getenv("GGML_RPC_QUIC_QLOG_DIR");
        if (qlog) qlog_dir = qlog;
        if (const char * timeout = std::getenv("GGML_RPC_QUIC_IO_TIMEOUT_MS")) {
            char * end = nullptr;
            long value = std::strtol(timeout, &end, 10);
            if (!end || *end || value < 100 || value > IO_TIMEOUT_MS) return false;
            timeout_ms = static_cast<int>(value);
        }
        if (const char * timeout = std::getenv("GGML_RPC_QUIC_IDLE_TIMEOUT_MS")) {
            char * end = nullptr;
            long value = std::strtol(timeout, &end, 10);
            if (!end || *end || value < 1000 || value > 600000) return false;
            idle_timeout_ms = static_cast<int>(value);
        }
        quic = picoquic_create(4, cert, key, ca, ALPN, &engine_t::callback,
                              this, nullptr, nullptr, nullptr, now_us(), nullptr,
                              nullptr, nullptr, 0);
        if (!quic) { GGML_LOG_ERROR("picoquic_create failed\n"); return false; }
        picoquic_set_client_cert_verification_policy(quic, 1);
        if (server) picoquic_set_client_authentication(quic, 1);
        static std::once_flag cc_registry_once;
        std::call_once(cc_registry_once, picoquic_register_all_congestion_control_algorithms);
        auto * alg = picoquic_get_congestion_algorithm(cc.c_str());
        if (!alg) { GGML_LOG_ERROR("unknown picoquic CC: %s\n", cc.c_str()); return false; }
        picoquic_set_default_congestion_algorithm(quic, alg);
        picoquic_set_cwin_max(quic, CWIN_BYTES);
        picoquic_set_max_data_control(quic, FLOW_BYTES);
        picoquic_tp_t tp = *picoquic_get_default_tp(quic);
        tp.initial_max_data = FLOW_BYTES;
        tp.initial_max_stream_data_bidi_local = FLOW_BYTES;
        tp.initial_max_stream_data_bidi_remote = FLOW_BYTES;
        tp.max_idle_timeout = idle_timeout_ms;
        tp.migration_disabled = 1;
        if (picoquic_set_default_tp(quic, &tp) != 0) { GGML_LOG_ERROR("QUIC transport parameters rejected\n"); return false; }
        if (!qlog_dir.empty() && picoquic_set_qlog(quic, qlog_dir.c_str()) != 0) { GGML_LOG_ERROR("QUIC qlog directory rejected\n"); return false; }
        wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd < 0) GGML_LOG_ERROR("QUIC eventfd failed: %s\n", strerror(errno));
        return wake_fd >= 0;
    }
    void start() { io_thread = std::thread([this]() { run(); }); }
    // prepare_next_packet may delete a disconnected server connection itself.
    // Reconcile pointer identities before any further dereference of our map.
    void reconcile_connections() {
        std::unordered_set<picoquic_cnx_t *> live;
        for (auto * p = picoquic_get_first_cnx(quic); p; p = picoquic_get_next_cnx(p)) live.insert(p);
        for (auto it = channels.begin(); it != channels.end();) {
            if (live.count(it->first) == 0) {
                it->second->failed = true;
                it->second->cnx = nullptr;
                it = channels.erase(it);
            } else ++it;
        }
    }
    static int callback(picoquic_cnx_t * cnx, uint64_t stream, uint8_t * bytes,
                        size_t length, picoquic_call_back_event_t event,
                        void * context, void *) {
        auto * e = static_cast<engine_t *>(context);
        auto it = e->channels.find(cnx);
        if (it == e->channels.end()) {
            if (!e->server || e->channels.size() >= 4) return -1;
            auto ch = std::make_shared<channel_t>();
            ch->cnx = cnx;
            ch->id = e->next_id++;
            it = e->channels.emplace(cnx, std::move(ch)).first;
        }
        channel_t & ch = *it->second;
        if (event == picoquic_callback_ready) {
            ch.ready = true;
            picoquic_enable_keep_alive(cnx, static_cast<uint64_t>(std::min(60000, e->idle_timeout_ms / 4)) * 1000);
            if (e->server && !ch.accepted) { ch.accepted = true; e->pending.push_back(it->second); }
        } else if (event == picoquic_callback_stream_data || event == picoquic_callback_stream_fin) {
            if (stream != 0 || length > ch.rx.bytes.size() - ch.rx.used) { ch.failed = true; return -1; }
            if (!ch.flow_manual) {
                if (picoquic_set_app_flow_control(cnx, 0, 1) != 0) { ch.failed = true; return -1; }
                ch.flow_manual = true;
            }
            ch.rx.write(bytes, length);
            if (event == picoquic_callback_stream_fin) ch.peer_fin = true;
        } else if (event == picoquic_callback_prepare_to_send) {
            if (stream != 0) return -1;
            size_t n = std::min(length, ch.tx.used);
            uint8_t * out = picoquic_provide_stream_data_buffer(bytes, n, 0, ch.tx.used > n);
            if (!out && n) return -1;
            if (n) { ch.tx.read(out, n); ch.app_sent += n; }
        } else if (event == picoquic_callback_stream_released) {
            if (stream == 0) ch.fin_released = true;
        } else if (event == picoquic_callback_close || event == picoquic_callback_application_close ||
                   event == picoquic_callback_stateless_reset || event == picoquic_callback_stream_reset) {
            if (!ch.ready) GGML_LOG_ERROR("QUIC handshake closed: event=%d local_error=%llu remote_error=%llu\n",
                                          static_cast<int>(event),
                                          static_cast<unsigned long long>(picoquic_get_local_error(cnx)),
                                          static_cast<unsigned long long>(picoquic_get_remote_error(cnx)));
            if (!ch.final_stats_written) {
                e->write_stats(&ch);
                ch.final_stats_written = true;
            }
            ch.failed = true;
        }
        return 0;
    }
    bool drive() {
        if (stopping) return false;
        last_progress = false;
        std::array<uint8_t, PICOQUIC_MAX_PACKET_SIZE> packet{};
        for (auto & kv : channels) {
            if (kv.second->tx.used && !kv.second->fin_requested)
                if (picoquic_mark_active_stream_v2(kv.first, 0, 1) != 0) kv.second->failed = true;
            if (kv.second->flow_update && kv.second->flow_manual && !kv.second->closing) {
                kv.second->flow_update = false;
                if (picoquic_open_flow_control(kv.first, 0, RING_BYTES - kv.second->rx.used) != 0)
                    kv.second->failed = true;
            }
        }
        for (int i = 0; i < 128; ++i) {
            size_t len = 0;
            sockaddr_storage to = {}, from = {};
            int if_index = 0;
            picoquic_connection_id_t cid = {};
            picoquic_cnx_t * last = nullptr;
            int prepared = picoquic_prepare_next_packet(quic, now_us(), packet.data(), packet.size(),
                                                        &len, &to, &from, &if_index, &cid, &last);
            if (prepared != 0) { GGML_LOG_ERROR("QUIC prepare packet failed: %d\n", prepared); return false; }
            reconcile_connections();
            if (len == 0) break;
            socklen_t slen = to.ss_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in);
            ssize_t sent = sendto(fd, packet.data(), len, 0, reinterpret_cast<sockaddr *>(&to), slen);
            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS) { ++udp_send_drops; continue; }
                GGML_LOG_ERROR("QUIC UDP send failed: %s\n", strerror(errno)); return false;
            }
            wire_sent += static_cast<uint64_t>(sent);
            last_progress = true;
        }
        {
            for (int i = 0; i < 128; ++i) {
                sockaddr_storage peer = {};
                socklen_t plen = sizeof(peer);
                ssize_t len = recvfrom(fd, packet.data(), packet.size(), MSG_DONTWAIT,
                                       reinterpret_cast<sockaddr *>(&peer), &plen);
                if (len < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    GGML_LOG_ERROR("QUIC UDP receive failed: %s\n", strerror(errno)); return false;
                }
                wire_received += len;
                last_progress = true;
                if (picoquic_incoming_packet(quic, packet.data(), len,
                                             reinterpret_cast<sockaddr *>(&peer),
                                             reinterpret_cast<sockaddr *>(&local), 0, 0, now_us()) != 0) ++udp_parse_errors;
                reconcile_connections();
            }
        }
        return true;
    }
    void run() {
        while (!stopping) {
            int timeout = 50;
            {
                std::lock_guard<std::mutex> lock(mu);
                for (auto & kv : channels) {
                    if (kv.second->closing && !kv.second->close_sent && kv.second->cnx) {
                        channel_t & ch = *kv.second;
                        const bool expired = now_us() - ch.close_started_us >= GRACEFUL_CLOSE_US;
                        if (!expired && !ch.failed && ch.ready && !ch.fin_requested && ch.tx.used == 0) {
                            if (picoquic_add_to_stream_with_ctx(kv.first, 0, nullptr, 0, 1, &ch) == 0) {
                                ch.fin_requested = true;
                            } else {
                                // A server may not yet have received the client-owned stream 0.
                                ch.failed = true;
                            }
                        }
                        // The server's released callback proves that both stream FINs
                        // arrived and its own FIN was acknowledged. The client waits
                        // for the server's connection close, which confirms that its
                        // ACK of the server FIN was observed.
                        if (expired || ch.failed || !ch.ready || (server && ch.fin_released)) {
                            if (!ch.final_stats_written) {
                                write_stats(&ch);
                                ch.final_stats_written = true;
                            }
                            (void)picoquic_close(kv.first, 0);
                            ch.close_sent = true;
                            if (expired && !server && ch.fin_released)
                                GGML_LOG_ERROR("QUIC client close confirmation timed out after stream release\n");
                            else if (expired && !ch.fin_released)
                                GGML_LOG_ERROR("QUIC graceful close timed out before stream release\n");
                        }
                    }
                }
                if (!drive()) { stopping = true; cv.notify_all(); break; }
                for (auto & kv : channels) {
                    kv.second->acked_empty = kv.second->tx.used == 0 &&
                        picoquic_is_cnx_backlog_empty(kv.first) != 0;
                }
                if (!stats_dir.empty() && now_us() - last_stats_us >= 5000000) {
                    for (auto & kv : channels) write_stats(kv.second.get());
                    last_stats_us = now_us();
                }
                const int64_t wake_us = picoquic_get_next_wake_delay(quic, now_us(), 50000);
                timeout = std::max(0, std::min(50, static_cast<int>((wake_us + 999) / 1000)));
                if (timeout == 0 && !last_progress) timeout = 1;
                cv.notify_all();
            }
            pollfd fds[2] = {{fd, POLLIN, 0}, {wake_fd, POLLIN, 0}};
            int rc = poll(fds, 2, timeout);
            if (rc < 0 && errno != EINTR) {
                GGML_LOG_ERROR("QUIC UDP poll failed: %s\n", strerror(errno));
                stopping = true; cv.notify_all(); break;
            }
            if (fds[1].revents & POLLIN) { uint64_t discard; (void)::read(wake_fd, &discard, sizeof(discard)); }
        }
        cv.notify_all();
    }
    bool wait_ready(const std::shared_ptr<channel_t> & ch) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
            return ch->ready || ch->failed || ch->closing || stopping;
        });
        return ch->ready && !ch->failed && !ch->closing;
    }
    void write_stats(channel_t * ch) {
        if (!ch || !ch->cnx || stats_dir.empty()) return;
        picoquic_path_quality_t q = {};
        picoquic_get_default_path_quality(ch->cnx, &q);
        const std::string path = stats_dir + "/rpc-quic-" + std::to_string(getpid()) + "-" +
                                 std::to_string(engine_id) + "-" + std::to_string(ch->id) + ".jsonl";
        std::ofstream out(path, std::ios::app);
        if (out && out.tellp() > 64 * 1024 * 1024) return;
        sockaddr * peer = nullptr;
        picoquic_get_peer_addr(ch->cnx, &peer);
        char peer_host[NI_MAXHOST] = {}, peer_port[NI_MAXSERV] = {};
        if (peer) getnameinfo(peer, peer->sa_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in),
                              peer_host, sizeof(peer_host), peer_port, sizeof(peer_port), NI_NUMERICHOST | NI_NUMERICSERV);
        sockaddr_storage selected_local = local;
        sockaddr_storage path_local = {};
        if (picoquic_get_path_addr(ch->cnx, 0, 1, &path_local) == 0 &&
            (path_local.ss_family == AF_INET || path_local.ss_family == AF_INET6)) selected_local = path_local;
        char local_host[NI_MAXHOST] = {}, local_port[NI_MAXSERV] = {};
        getnameinfo(reinterpret_cast<sockaddr *>(&selected_local), selected_local.ss_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in),
                    local_host, sizeof(local_host), local_port, sizeof(local_port), NI_NUMERICHOST | NI_NUMERICSERV);
        const auto utc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (out) out << "{\"schema\":\"hoshino-rpc-quic-v1\",\"utc_ns\":" << utc_ns
                     << ",\"mono_us\":" << now_us() << ",\"engine_id\":\"" << getpid() << "-" << engine_id
                     << "\",\"udp_payload_scope\":\"engine\",\"cc\":\"" << cc
                     << "\",\"local\":\"" << local_host << ":" << local_port
                     << "\",\"peer\":\"" << peer_host << ":" << peer_port
                     << "\",\"app_sent\":" << ch->app_sent
                     << ",\"app_received\":" << ch->app_received
                     << ",\"udp_payload_sent_engine\":" << wire_sent
                     << ",\"udp_payload_received_engine\":" << wire_received
                     << ",\"udp_send_drops_engine\":" << udp_send_drops
                     << ",\"udp_parse_errors_engine\":" << udp_parse_errors
                     << ",\"socket_receive_buffer_effective\":" << socket_receive_buffer
                     << ",\"socket_send_buffer_effective\":" << socket_send_buffer
                     << ",\"quic_sent\":" << picoquic_get_data_sent(ch->cnx)
                     << ",\"quic_received\":" << picoquic_get_data_received(ch->cnx)
                     << ",\"rtt_us\":" << q.rtt << ",\"rtt_min_us\":" << q.rtt_min
                     << ",\"cwnd_bytes\":" << q.cwin << ",\"pacing_bytes_per_s\":" << q.pacing_rate
                     << ",\"packets_sent\":" << q.sent << ",\"packets_lost\":" << q.lost
                     << ",\"timer_losses\":" << q.timer_losses
                     << ",\"spurious_losses\":" << q.spurious_losses << "}\n";
    }
};

bool resolve_numeric(const char * host, int port, sockaddr_storage & out, socklen_t & len) {
    if (!host || port < 1 || port > 65535) return false;
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV;
    addrinfo * result = nullptr;
    if (getaddrinfo(host, std::to_string(port).c_str(), &hints, &result) != 0) return false;
    if (!result || result->ai_addrlen > sizeof(out)) { if (result) freeaddrinfo(result); return false; }
    memcpy(&out, result->ai_addr, result->ai_addrlen);
    len = result->ai_addrlen;
    freeaddrinfo(result);
    return true;
}
} // namespace

struct socket_t::impl {
    std::shared_ptr<engine_t> engine;
    std::shared_ptr<channel_t> channel;
    bool listener = false;
    impl(std::shared_ptr<engine_t> e, std::shared_ptr<channel_t> c, bool l) : engine(std::move(e)), channel(std::move(c)), listener(l) {}
};

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}
socket_t::~socket_t() {
    close();
}
void socket_t::close() {
    if (!pimpl || !pimpl->engine) return;
    auto & e = *pimpl->engine;
    std::unique_lock<std::mutex> lock(e.mu);
    if (pimpl->listener) {
        for (auto & kv : e.channels) {
            kv.second->closing = true;
            if (!kv.second->close_started_us) kv.second->close_started_us = now_us();
        }
    }
    if (pimpl->channel && !pimpl->channel->closing) {
        pimpl->channel->closing = true;
        pimpl->channel->close_started_us = now_us();
    }
    e.cv.notify_all();
    e.wake();
    if (pimpl->channel) e.cv.wait_for(lock, std::chrono::seconds(12), [&]() {
        return pimpl->channel->close_sent || e.stopping || !pimpl->channel->cnx;
    });
    if (pimpl->listener) {
        e.cv.wait_for(lock, std::chrono::seconds(2), [&]() {
            for (auto & kv : e.channels) if (!kv.second->close_sent) return false;
            return true;
        });
        e.stopping = true;
        e.wake();
    }
}

socket_ptr socket_t::create_server(const char * host, int port) {
    sockaddr_storage addr = {};
    socklen_t len = 0;
    if (!resolve_numeric(host, port, addr, len)) return nullptr;
    auto e = std::make_shared<engine_t>();
    if (!e->init(true)) return nullptr;
    e->fd = socket(addr.ss_family, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (e->fd < 0 || !e->configure_socket_buffers() ||
        bind(e->fd, reinterpret_cast<sockaddr *>(&addr), len) != 0) {
        GGML_LOG_ERROR("QUIC UDP bind failed: %s\n", strerror(errno)); return nullptr;
    }
    e->local = addr;
    e->start();
    return socket_ptr(new socket_t(std::unique_ptr<impl>(new impl(e, nullptr, true))));
}

socket_ptr socket_t::connect(const char * host, int port) {
    sockaddr_storage addr = {};
    socklen_t len = 0;
    if (!resolve_numeric(host, port, addr, len)) return nullptr;
    auto e = std::make_shared<engine_t>();
    if (!e->init(false)) return nullptr;
    e->fd = socket(addr.ss_family, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (e->fd < 0 || !e->configure_socket_buffers()) return nullptr;
    if (addr.ss_family == AF_INET) {
        sockaddr_in local = {}; local.sin_family = AF_INET;
        if (bind(e->fd, reinterpret_cast<sockaddr *>(&local), sizeof(local)) != 0) return nullptr;
    } else {
        sockaddr_in6 local = {}; local.sin6_family = AF_INET6;
        if (bind(e->fd, reinterpret_cast<sockaddr *>(&local), sizeof(local)) != 0) return nullptr;
    }
    if (::connect(e->fd, reinterpret_cast<sockaddr *>(&addr), len) != 0) return nullptr;
    socklen_t llen = sizeof(e->local);
    if (getsockname(e->fd, reinterpret_cast<sockaddr *>(&e->local), &llen) != 0) return nullptr;
    const char * sni = std::getenv("GGML_RPC_QUIC_SNI");
    if (!sni || !*sni) sni = DEFAULT_SNI;
    auto * cnx = picoquic_create_client_cnx(e->quic, reinterpret_cast<sockaddr *>(&addr), now_us(),
                                             0, sni, ALPN, &engine_t::callback, e.get());
    if (!cnx) return nullptr;
    auto ch = std::make_shared<channel_t>(); ch->cnx = cnx;
    ch->id = e->next_id++;
    e->channels.emplace(cnx, ch);
    e->start();
    if (!e->wait_ready(ch)) return nullptr;
    return socket_ptr(new socket_t(std::unique_ptr<impl>(new impl(e, ch, false))));
}

socket_ptr socket_t::accept() {
    if (!pimpl || !pimpl->listener) return nullptr;
    auto & e = *pimpl->engine;
    std::unique_lock<std::mutex> lock(e.mu);
    while (!e.stopping) {
        if (!e.pending.empty()) {
            auto ch = e.pending.front(); e.pending.pop_front();
            if (!ch->failed && ch->ready) return socket_ptr(new socket_t(std::unique_ptr<impl>(new impl(pimpl->engine, ch, false))));
        }
        e.cv.wait(lock);
    }
    return nullptr;
}

bool socket_t::send_data(const void * data, size_t size) {
    if (!pimpl || !pimpl->channel || (!data && size)) return false;
    auto & e = *pimpl->engine;
    auto & ch = *pimpl->channel;
    auto * src = static_cast<const uint8_t *>(data);
    std::unique_lock<std::mutex> lock(e.mu);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(e.timeout_ms);
    while (size && !ch.failed && !ch.closing && !e.stopping) {
        size_t n = ch.tx.write(src, size);
        if (n) ch.acked_empty = false;
        src += n; size -= n;
        if (n) e.wake();
        if (size && n == 0) {
            if (!e.cv.wait_until(lock, deadline, [&]() {
                return ch.tx.used < ch.tx.bytes.size() || ch.failed || ch.closing || e.stopping;
            })) return false;
        }
        if (n) deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(e.timeout_ms);
    }
    return !ch.failed && !ch.closing && !e.stopping;
}

bool socket_t::recv_data(void * data, size_t size) {
    if (!pimpl || !pimpl->channel || (!data && size)) return false;
    auto & e = *pimpl->engine;
    auto & ch = *pimpl->channel;
    auto * dst = static_cast<uint8_t *>(data);
    std::unique_lock<std::mutex> lock(e.mu);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(e.timeout_ms);
    while (size && !ch.closing && !e.stopping) {
        size_t n = ch.rx.read(dst, size);
        ch.app_received += n; dst += n; size -= n;
        if (n) deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(e.timeout_ms);
        if (size && (ch.failed || ch.peer_fin)) return false;
        if (n) { ch.flow_update = true; e.wake(); }
        if (size && e.cv.wait_until(lock, deadline) == std::cv_status::timeout) return false;
    }
    return size == 0;
}
bool socket_t::wait_readable() {
    if (!pimpl || !pimpl->channel) return false;
    auto & e = *pimpl->engine;
    auto & ch = *pimpl->channel;
    std::unique_lock<std::mutex> lock(e.mu);
    e.cv.wait(lock, [&]() {
        return ch.rx.used != 0 || ch.failed || ch.peer_fin || ch.closing || e.stopping;
    });
    return ch.rx.used != 0 && !ch.closing && !e.stopping;
}
bool socket_t::flush() {
    if (!pimpl || !pimpl->channel) return false;
    auto & e = *pimpl->engine;
    auto & ch = *pimpl->channel;
    std::unique_lock<std::mutex> lock(e.mu);
    e.wake();
    return e.cv.wait_for(lock, std::chrono::milliseconds(e.timeout_ms), [&]() {
        return ch.tx.used == 0 || ch.failed || ch.closing || e.stopping;
    }) && !ch.failed && !ch.closing && !e.stopping;
}
bool socket_t::drain_acked() {
    if (!pimpl || !pimpl->channel) return false;
    auto & e = *pimpl->engine;
    auto & ch = *pimpl->channel;
    std::unique_lock<std::mutex> lock(e.mu);
    e.wake();
    return e.cv.wait_for(lock, std::chrono::milliseconds(e.timeout_ms), [&]() {
        return ch.acked_empty || ch.failed || ch.closing || e.stopping;
    }) && !ch.failed && !ch.closing && !e.stopping;
}
void socket_t::get_caps(uint8_t * caps) { memset(caps, 0, RPC_CONN_CAPS_SIZE); }
void socket_t::update_caps(const uint8_t *) {}
bool rpc_transport_init() { return true; }
void rpc_transport_shutdown() {}
