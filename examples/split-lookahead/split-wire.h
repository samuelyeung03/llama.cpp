#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace split_wire {

constexpr uint16_t kVersion = 1;
constexpr uint32_t kEarlyLookahead = 1;
constexpr uint32_t kMaxFrameBytes = 256U << 20;
constexpr uint64_t kNoId = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kNoToken = std::numeric_limits<uint32_t>::max();

enum class Type : uint16_t { Hello = 1, HelloAck = 2, Activation = 3, Reply = 4, Close = 5 };
enum class Decision : uint8_t { Ordinary = 0, Accepted = 1, Rejected = 2 };

struct ProtocolError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Handshake {
    std::array<uint8_t, 32> model_hash{};
    std::array<uint8_t, 32> tokenizer_hash{};
    uint32_t split_layer = 0;
    uint32_t total_layers = 0;
    uint32_t hidden_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t max_tokens = 0;
    uint32_t capabilities = 0;
};

struct Frame {
    Type type = Type::Close;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    uint64_t frame_id = 0;
    uint64_t token_pos = 0;
    uint32_t token_count = 0;
    uint64_t candidate_id = kNoId;
    uint32_t token_id = kNoToken;
    uint64_t predecessor_frame_id = kNoId;
    uint64_t predecessor_pos = kNoId;
    uint32_t predecessor_target_token = kNoToken;
    uint32_t target_token = kNoToken;
    Decision decision = Decision::Ordinary;
    Handshake handshake{};
    std::vector<uint16_t> activation_f16;
};

std::vector<uint8_t> encode_frame(const Frame & frame);
Frame decode_frame(const std::vector<uint8_t> & bytes);
void send_frame(int fd, const Frame & frame);
Frame recv_frame(int fd);
void validate_handshake(const Handshake & expected, const Handshake & actual);
void validate_reply(const Frame & activation, const Frame & reply, Decision expected_decision, uint32_t expected_predecessor_target = kNoToken);

class WorkerState {
public:
    explicit WorkerState(Handshake expected);
    void start(uint64_t request_id, uint64_t session_id);
    void reset();
    Decision inspect_activation(const Frame & frame);
    void complete_activation(const Frame & frame, uint32_t target_token);
    uint64_t next_frame_id() const { return next_frame_id_; }
    uint64_t next_token_pos() const { return next_token_pos_; }
    uint32_t retained_target_token() const { return retained_target_; }

private:
    Handshake expected_;
    uint64_t request_id_ = 0;
    uint64_t session_id_ = 0;
    uint64_t next_frame_id_ = 0;
    uint64_t next_token_pos_ = 0;
    uint64_t next_candidate_id_ = 0;
    uint64_t pending_frame_id_ = kNoId;
    uint64_t pending_pos_ = kNoId;
    uint64_t pending_candidate_id_ = kNoId;
    uint32_t pending_count_ = 0;
    uint64_t retained_frame_id_ = kNoId;
    uint64_t retained_pos_ = kNoId;
    uint32_t retained_target_ = kNoToken;
    bool active_ = false;
};

} // namespace split_wire
