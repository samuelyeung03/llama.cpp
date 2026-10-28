#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef GGML_RPC_QUIC
struct rpc_quic_stats_t {
    uint64_t udp_sent_engine = 0;
    uint64_t udp_received_engine = 0;
    uint64_t app_sent = 0;
    uint64_t app_received = 0;
    uint64_t udp_send_drops_engine = 0;
    uint64_t udp_parse_errors_engine = 0;
    uint32_t socket_receive_buffer = 0;
    uint32_t socket_send_buffer = 0;
    uint16_t local_port = 0;
    uint16_t peer_port = 0;
    uint32_t failed = 0;
    char local_ipv4[16] = {};
    char peer_ipv4[16] = {};
};
#endif

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();
#ifdef GGML_RPC_QUIC
    void close(); // Graceful close; cancels blocked application I/O.
    void shutdown(); // Abort owned I/O without waiting for peer FIN.
    bool drain_acked(); // Bounded wait for queued data and QUIC retransmission backlog.
    bool wait_readable(); // Between complete RPC messages only; payload reads remain timed.
    bool get_quic_stats(rpc_quic_stats_t * stats); // UDP counters cover the shared engine.
#endif

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
