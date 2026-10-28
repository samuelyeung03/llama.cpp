# H3 native QUIC carrier

This standalone C ABI reuses the fork's existing Picoquic `socket_t` engine.
It builds no model, CUDA backend, or inference runner. Python can perform tile
compression and framing with bounded bulk calls while packet receive, ACK,
pacing, retransmission and connection callbacks run entirely in C++.
The default H3 ALPN is `hoshino-h3-native/1`; callers pass it to `h3q_configure`.
This is an opt-in carrier, not a performance or inference qualification.

## Build from pinned local dependencies

Use the exact Picoquic and Picotls source pins in the parent repository's
`ports/native-variants.json`. Do not fetch or advance dependencies implicitly.

```sh
cmake -S examples/h3-native-quic -B /path/to/owned-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPICOQUIC_SOURCE_DIR=/path/to/pinned-picoquic \
  -DFETCHCONTENT_SOURCE_DIR_PICOTLS=/path/to/pinned-picotls
cmake --build /path/to/owned-build --target h3-native-quic
```

The output is `libh3-native-quic.so`. Inspect its runtime dependencies and hash
its exact bytes before use. Symbols are hidden except the declared `h3q_*`
exports; linked static archives are excluded from symbol export. This avoids
exporting dependency internals into the Python process. Dependency selection
and any static OpenSSL build must be recorded separately for each platform.

## ABI version 1

See `h3-quic.h` for the exact C declarations and naturally aligned stats struct.
Opaque handles are unsigned 64-bit registry identities, not C++ pointers.
Every function except the version query returns a `h3q_status`: zero succeeds.
Native exceptions become `H3Q_INTERNAL`; exception text and credentials are not
returned through the ABI.

1. Call `h3q_configure(cert, key, ca, sni, cc, alpn, timeout_ms)` before opening
   a handle. Certificate paths must be absolute readable regular files. The
   private key must be owner-only, owned by the effective UID, and have one
   link. Leaf symlinks are rejected. An identical validated configuration is a no-op while handles or connection
   creations remain live; a different configuration returns `H3Q_BUSY`.
2. Use numeric, non-wildcard IPv4 addresses. `h3q_listen` accepts port zero;
   obtain its actual port through `h3q_get_stats`. `h3q_connect` requires a
   nonzero port. Accept on a listener to obtain a connection handle.
3. `h3q_send_exact` copies a caller buffer into the bounded native TX ring;
   return does not mean remote delivery. `h3q_recv_exact` fills the supplied
   writable buffer or reports failure. Each bulk call is at most 8 MiB. Both
   directions permit empty calls with a null buffer. The caller must keep
   buffers alive and stable until the call returns.
4. One accept/send/receive/flush/drain operation is allowed per handle. A
   concurrent application operation returns `H3Q_BUSY`. Stats and shutdown
   may run concurrently. Different handles can run concurrently.
5. `h3q_flush` waits until the application TX ring drains into QUIC. It is not
   delivery confirmation. `h3q_drain_acked` additionally waits for the QUIC
   retransmission backlog; use it at terminal delivery boundaries, not after
   every tile or packet. Application framing must independently verify the
   decoded tensor and final full-activation hash.
6. `h3q_shutdown` immediately aborts owned application I/O without waiting for
   peer FIN or delivery confirmation. Use it for cancellation and failure. The
   separate `h3q_close` performs the existing bounded graceful FIN/close; drain
   acknowledged bytes first when terminal delivery must be confirmed. Both
   invalidate new bulk work and retain the handle for stats and lifecycle
   reconciliation. Close after shutdown returns promptly. Shutting down a
   listener stops its shared engine and wakes blocked accepts and connections.
   Abort is a local cleanup action, not proof that a remote close was received.
7. Join every owned native call before `h3q_free`. Free returns `H3Q_BUSY` if
   calls still hold a lease, and rejects a handle not previously shut down.
   Free removes the registry entry; stale handles return `H3Q_BAD_HANDLE`.

`h3q_accept` waits until an authenticated connection arrives or the listener is
shut down. A connect handshake has the configured native timeout. Stream I/O
uses the engine's per-progress timeout, not a whole-transfer deadline. An async
Python adapter must impose its own whole-operation deadline, shut down the
owned handle on cancellation, and join its worker before releasing buffers or
freeing handles. A connecting call must be joined through its native deadline
because its handle is returned only after the handshake.

## Authentication and configuration ownership

TLS certificate validation is preserved in both directions. The listener
requires a CA-verified client certificate; the client verifies the server's
CA and configured SNI. No verification-disable flag or plaintext socket fallback
is exposed. Keep the CA private signing key off the endpoints.

`h3q_configure` sets the existing engine's `GGML_RPC_QUIC_CERT`, `KEY`, `CA`,
`SNI`, `CC`, `ALPN` and `IO_TIMEOUT_MS` environment options. Configuration is
process-scoped and serialized by the ABI registry. Do not mutate these options
externally, replace configured credential files, or invoke a separate copy of
this engine directly while ABI handles are live. Use separate processes for
independent TLS configurations. `cc` accepts `newreno`, `cubic` or `bbr`;
`timeout_ms` accepts 100 through 300000.

The optional `GGML_RPC_QUIC_ALPN` hook is used by both server and client engine
creation. When unset, existing RPC callers retain `hoshino-rpc/1`. This carrier
does not add application authentication bytes; a Python H3 adapter can layer
its existing per-run challenge and message protocol above the encrypted stream.

## Bounds and counters

The existing engine uses an 8 MiB TX ring, an 8 MiB RX ring, application-managed
receive credit, initial 4 MiB QUIC flow credit, an 8 MiB congestion-window cap
and at most four connections per listener. It requests 4 MiB socket-local send
and receive buffers; the kernel may cap or double effective values. No sysctl,
route, firewall, offload or machine-wide policy is changed.

The 8 MiB call and ring limits are not whole-process memory limits. Caller
payloads, codec workspaces, concurrent handles and QUIC state need separate
owned-process admission and memory caps. Application framing must reject
oversized headers, encoded tiles and decoded lengths before allocation. This
ABI transports bytes and does not validate tensor metadata or decompress data.

`h3q_get_stats` requires `struct_bytes == sizeof(h3q_stats)` and fills the ABI
version and size. Its values are copied under the engine mutex. UDP counters
cover the entire engine, include encrypted QUIC headers and retransmissions,
and exclude UDP/IP/link headers. A listener and all its accepted connections
share these engine counters; never add their snapshots together. Application
bytes are per connection: send counts bytes consumed from its TX ring and
receive counts bytes returned to the caller. Effective socket buffers and
numeric local/peer addresses are reported separately. Listener peer fields
are empty. The failure field reports engine/channel failure state, not a
verified application completion.

Qualify exact bytes, authentication rejection, bad handles and lengths,
backpressure, cancellation, peer loss, reconnect and terminal ACK durability
before a separately authorized LAN comparison. Keep profiling separate from
benchmark timers and record exact source, library, runtime and input hashes.
