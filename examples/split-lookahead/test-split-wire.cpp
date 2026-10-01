#include "split-wire.h"

#include <stdexcept>
#include <functional>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace split_wire;

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error("check failed: " #expr); } while (0)

static void rejects(const std::function<void()> & fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const ProtocolError &) {
        rejected = true;
    }
    CHECK(rejected);
}

static Handshake handshake() {
    Handshake h;
    h.model_hash.fill(0x42);
    h.tokenizer_hash.fill(0x24);
    h.split_layer = 40;
    h.total_layers = 64;
    h.hidden_dim = 4;
    h.vocab_size = 100;
    h.max_tokens = 16;
    h.capabilities = kEarlyLookahead;
    return h;
}

static Frame ordinary(uint64_t frame_id, uint64_t pos) {
    Frame f;
    f.type = Type::Activation;
    f.request_id = 7;
    f.session_id = 9;
    f.frame_id = frame_id;
    f.token_pos = pos;
    f.token_count = 1;
    f.activation_f16 = {0x3c00, 0x4000, 0x4200, 0x4400};
    return f;
}

static Frame candidate(uint64_t frame_id, uint64_t pos, uint64_t candidate_id, uint32_t token) {
    Frame f = ordinary(frame_id, pos);
    f.candidate_id = candidate_id;
    f.token_id = token;
    f.predecessor_frame_id = frame_id == 1 ? 0 : frame_id - 1;
    f.predecessor_pos = pos - 1;
    return f;
}

static void test_round_trip_and_handshake() {
    Frame hello;
    hello.type = Type::Hello;
    hello.request_id = 7;
    hello.session_id = 9;
    hello.handshake = handshake();
    const Frame parsed = decode_frame(encode_frame(hello));
    CHECK(parsed.type == Type::Hello && parsed.handshake.model_hash == hello.handshake.model_hash);
    validate_handshake(handshake(), parsed.handshake);
    Handshake wrong = parsed.handshake;
    wrong.tokenizer_hash[0] ^= 1;
    rejects([&] { validate_handshake(handshake(), wrong); });
    wrong = parsed.handshake;
    wrong.hidden_dim++;
    rejects([&] { validate_handshake(handshake(), wrong); });
    wrong = parsed.handshake;
    wrong.capabilities = 0;
    rejects([&] { validate_handshake(handshake(), wrong); });

    const Frame activation = decode_frame(encode_frame(ordinary(0, 0)));
    CHECK(activation.activation_f16 == ordinary(0, 0).activation_f16);
    Frame reply;
    reply.type = Type::Reply;
    reply.request_id = 7;
    reply.session_id = 9;
    reply.frame_id = 0;
    reply.target_token = 12;
    CHECK(decode_frame(encode_frame(reply)).target_token == 12);
    validate_reply(activation, reply, Decision::Ordinary);
    reply.frame_id = 1;
    rejects([&] { validate_reply(activation, reply, Decision::Ordinary); });
}

static void test_bad_frames() {
    const std::vector<uint8_t> valid = encode_frame(ordinary(0, 0));
    for (size_t size = 0; size < valid.size(); ++size) {
        rejects([&] { decode_frame(std::vector<uint8_t>(valid.begin(), valid.begin() + size)); });
    }
    std::vector<uint8_t> bad = valid;
    bad.back() ^= 1;
    rejects([&] { decode_frame(bad); });
    bad = valid;
    bad[4] = 2;
    rejects([&] { decode_frame(bad); });
    bad = valid;
    bad[8] = bad[9] = bad[10] = bad[11] = 0xff;
    rejects([&] { decode_frame(bad); });
    Frame malformed = ordinary(0, 0);
    malformed.candidate_id = 0;
    rejects([&] { encode_frame(malformed); });
}

static void test_socket_fragment_and_truncation() {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    const std::vector<uint8_t> bytes = encode_frame(ordinary(0, 0));
    CHECK(write(sockets[0], bytes.data(), 3) == 3);
    CHECK(write(sockets[0], bytes.data() + 3, bytes.size() - 3) == ssize_t(bytes.size() - 3));
    CHECK(recv_frame(sockets[1]).activation_f16.size() == 4);
    send_frame(sockets[0], ordinary(1, 1));
    CHECK(recv_frame(sockets[1]).frame_id == 1);
    CHECK(write(sockets[0], bytes.data(), bytes.size() - 1) == ssize_t(bytes.size() - 1));
    close(sockets[0]);
    rejects([&] { recv_frame(sockets[1]); });
    close(sockets[1]);
}

static void test_tcp_malformed_reply() {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(listener >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
    CHECK(listen(listener, 1) == 0);
    socklen_t address_length = sizeof(address);
    CHECK(getsockname(listener, reinterpret_cast<sockaddr *>(&address), &address_length) == 0);
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(client >= 0);
    CHECK(connect(client, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
    const int worker = accept(listener, nullptr, nullptr);
    CHECK(worker >= 0);

    const Frame proposal = candidate(1, 1, 0, 30);
    Frame wrong_ack;
    wrong_ack.type = Type::Reply;
    wrong_ack.request_id = proposal.request_id;
    wrong_ack.session_id = proposal.session_id;
    wrong_ack.frame_id = proposal.frame_id;
    wrong_ack.token_pos = proposal.token_pos;
    wrong_ack.candidate_id = proposal.candidate_id;
    wrong_ack.decision = Decision::Accepted;
    wrong_ack.target_token = 13;
    wrong_ack.predecessor_target_token = 12;
    send_frame(worker, wrong_ack);
    const Frame received = recv_frame(client);
    rejects([&] { validate_reply(proposal, received, Decision::Rejected, 12); });
    std::vector<uint8_t> corrupt = encode_frame(wrong_ack);
    corrupt.back() ^= 1;
    CHECK(send(worker, corrupt.data(), corrupt.size(), 0) == ssize_t(corrupt.size()));
    rejects([&] { recv_frame(client); });
    close(worker);
    close(client);
    close(listener);
}

static void test_ordering_and_rejected_candidate() {
    WorkerState state(handshake());
    state.start(7, 9);
    Frame first = ordinary(0, 0);
    rejects([&] { state.inspect_activation(ordinary(1, 0)); });
    CHECK(state.inspect_activation(first) == Decision::Ordinary);
    rejects([&] { state.inspect_activation(ordinary(1, 1)); });
    state.complete_activation(first, 12);
    CHECK(state.next_frame_id() == 1 && state.next_token_pos() == 1);

    Frame rejected = candidate(1, 1, 0, 30);
    CHECK(state.inspect_activation(rejected) == Decision::Rejected);
    CHECK(state.next_frame_id() == 2 && state.next_token_pos() == 1);
    CHECK(state.retained_target_token() == 12);
    Frame reject_reply;
    reject_reply.type = Type::Reply;
    reject_reply.request_id = 7;
    reject_reply.session_id = 9;
    reject_reply.frame_id = 1;
    reject_reply.token_pos = 1;
    reject_reply.candidate_id = 0;
    reject_reply.decision = Decision::Rejected;
    reject_reply.target_token = 12;
    reject_reply.predecessor_target_token = 12;
    validate_reply(rejected, reject_reply, Decision::Rejected, 12);
    rejects([&] { validate_reply(rejected, reject_reply, Decision::Accepted, 12); });
    rejects([&] { validate_reply(rejected, reject_reply, Decision::Rejected, 13); });
    rejects([&] { state.complete_activation(rejected, 30); });
    rejects([&] { state.inspect_activation(ordinary(1, 1)); });
    Frame corrected = ordinary(2, 1);
    CHECK(state.inspect_activation(corrected) == Decision::Ordinary);
    state.complete_activation(corrected, 12);
    CHECK(state.next_token_pos() == 2);

    Frame bad_predecessor = candidate(3, 2, 1, 12);
    bad_predecessor.predecessor_frame_id = 1;
    rejects([&] { state.inspect_activation(bad_predecessor); });
    Frame accepted = candidate(3, 2, 1, 12);
    CHECK(state.inspect_activation(accepted) == Decision::Accepted);
    Frame altered = accepted;
    altered.token_count = 2;
    rejects([&] { state.complete_activation(altered, 13); });
    state.complete_activation(accepted, 13);
    CHECK(state.next_frame_id() == 4 && state.next_token_pos() == 3);
    rejects([&] { state.inspect_activation(accepted); });

    state.start(8, 10);
    CHECK(state.next_frame_id() == 0 && state.next_token_pos() == 0);
    rejects([&] { state.inspect_activation(first); });
}

int main() {
    test_round_trip_and_handshake();
    test_bad_frames();
    test_socket_fragment_and_truncation();
    test_tcp_malformed_reply();
    test_ordering_and_rejected_candidate();
    std::cout << "split wire tests passed\n";
}
