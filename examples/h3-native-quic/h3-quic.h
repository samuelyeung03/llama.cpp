#ifndef H3_QUIC_H
#define H3_QUIC_H
#include <stdint.h>
#if defined(__GNUC__)
#define H3Q_API __attribute__((visibility("default")))
#else
#define H3Q_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define H3Q_ABI_VERSION 1u
#define H3Q_MAX_CALL_BYTES (8u * 1024u * 1024u)
typedef uint64_t h3q_handle;
enum h3q_status {
    H3Q_OK = 0,
    H3Q_INVALID = 1,
    H3Q_BAD_HANDLE = 2,
    H3Q_BUSY = 3,
    H3Q_CLOSED = 4,
    H3Q_IO_ERROR = 5,
    H3Q_INTERNAL = 6
};
typedef struct h3q_stats {
    uint32_t abi_version;
    uint32_t struct_bytes;
    uint64_t udp_sent_engine;
    uint64_t udp_received_engine;
    uint64_t app_sent;
    uint64_t app_received;
    uint64_t udp_send_drops_engine;
    uint64_t udp_parse_errors_engine;
    uint32_t socket_receive_buffer;
    uint32_t socket_send_buffer;
    uint16_t local_port;
    uint16_t peer_port;
    uint32_t failed;
    char local_ipv4[16];
    char peer_ipv4[16];
} h3q_stats;
H3Q_API uint32_t h3q_abi_version(void);
H3Q_API int h3q_configure(const char * cert, const char * key, const char * ca,
                         const char * sni, const char * cc, const char * alpn,
                         uint32_t timeout_ms);
H3Q_API int h3q_listen(const char * ipv4, uint16_t port, h3q_handle * out);
H3Q_API int h3q_connect(const char * ipv4, uint16_t port, h3q_handle * out);
H3Q_API int h3q_accept(h3q_handle listener, h3q_handle * out);
H3Q_API int h3q_send_exact(h3q_handle handle, const void * data, uint64_t bytes);
H3Q_API int h3q_recv_exact(h3q_handle handle, void * data, uint64_t bytes);
H3Q_API int h3q_flush(h3q_handle handle);
H3Q_API int h3q_drain_acked(h3q_handle handle);
H3Q_API int h3q_shutdown(h3q_handle handle);
H3Q_API int h3q_close(h3q_handle handle);
H3Q_API int h3q_free(h3q_handle handle);
H3Q_API int h3q_get_stats(h3q_handle handle, h3q_stats * out, uint32_t struct_bytes);
#ifdef __cplusplus
}
#endif
#endif
