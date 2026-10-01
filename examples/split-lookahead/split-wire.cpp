#include "split-wire.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <sys/socket.h>

namespace split_wire {
namespace {

constexpr uint32_t kMagic = 0x31574c53; // SLW1 in little-endian wire order
constexpr size_t kHeaderBytes = 16;

void put8(std::vector<uint8_t> & out, uint8_t value) { out.push_back(value); }
void put16(std::vector<uint8_t> & out, uint16_t value) {
    for (int i = 0; i < 2; ++i) put8(out, uint8_t(value >> (i * 8)));
}
void put32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) put8(out, uint8_t(value >> (i * 8)));
}
void put64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) put8(out, uint8_t(value >> (i * 8)));
}

class Reader {
public:
    Reader(const uint8_t * data, size_t size) : data_(data), size_(size) {}
    uint8_t u8() {
        need(1);
        return data_[offset_++];
    }
    uint16_t u16() {
        const uint16_t low = u8();
        const uint16_t high = u8();
        return low | uint16_t(high << 8);
    }
    uint32_t u32() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= uint32_t(u8()) << (i * 8);
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) value |= uint64_t(u8()) << (i * 8);
        return value;
    }
    void copy(uint8_t * destination, size_t count) {
        need(count);
        std::memcpy(destination, data_ + offset_, count);
        offset_ += count;
    }
    size_t remaining() const { return size_ - offset_; }
private:
    void need(size_t count) const {
        if (count > size_ - offset_) throw ProtocolError("truncated split wire frame");
    }
    const uint8_t * data_;
    size_t size_;
    size_t offset_ = 0;
};

uint32_t crc32(const uint8_t * data, size_t size) {
    uint32_t crc = ~uint32_t(0);
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < values.size(); ++i) {
            uint32_t value = i;
            for (int j = 0; j < 8; ++j) value = (value >> 1) ^ ((value & 1) ? 0xedb88320U : 0);
            values[i] = value;
        }
        return values;
    }();
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

void validate_type(Type type) {
    if (type != Type::Hello && type != Type::HelloAck && type != Type::Activation && type != Type::Reply && type != Type::Close) {
        throw ProtocolError("unknown split wire message type");
    }
}

void validate_frame_fields(const Frame & frame) {
    validate_type(frame.type);
    if (frame.request_id == 0 || frame.session_id == 0) throw ProtocolError("zero request or session ID");
    if (frame.type != Type::Activation && !frame.activation_f16.empty()) throw ProtocolError("unexpected activation payload");
    if (frame.type == Type::Activation) {
        if (frame.token_count == 0 || frame.activation_f16.empty()) throw ProtocolError("empty activation");
        if (frame.decision != Decision::Ordinary || frame.target_token != kNoToken) throw ProtocolError("activation carries a reply decision");
        if (frame.candidate_id == kNoId) {
            if (frame.predecessor_frame_id != kNoId || frame.predecessor_pos != kNoId || frame.predecessor_target_token != kNoToken) {
                throw ProtocolError("ordinary activation carries candidate fields");
            }
        } else if (frame.token_count != 1 || frame.token_id == kNoToken || frame.predecessor_frame_id == kNoId || frame.predecessor_pos == kNoId || frame.predecessor_target_token != kNoToken) {
            throw ProtocolError("incomplete early candidate");
        }
    }
    if (frame.type == Type::Reply) {
        if (frame.token_count != 0 || frame.token_id != kNoToken || frame.target_token == kNoToken) throw ProtocolError("malformed reply");
        if (frame.decision != Decision::Ordinary && frame.decision != Decision::Accepted && frame.decision != Decision::Rejected) throw ProtocolError("unknown reply decision");
        if ((frame.candidate_id == kNoId) != (frame.decision == Decision::Ordinary)) throw ProtocolError("reply candidate decision mismatch");
    }
}

void put_handshake(std::vector<uint8_t> & out, const Handshake & hello) {
    out.insert(out.end(), hello.model_hash.begin(), hello.model_hash.end());
    out.insert(out.end(), hello.tokenizer_hash.begin(), hello.tokenizer_hash.end());
    put32(out, hello.split_layer);
    put32(out, hello.total_layers);
    put32(out, hello.hidden_dim);
    put32(out, hello.vocab_size);
    put32(out, hello.max_tokens);
    put32(out, hello.capabilities);
}

Handshake get_handshake(Reader & reader) {
    Handshake hello;
    reader.copy(hello.model_hash.data(), hello.model_hash.size());
    reader.copy(hello.tokenizer_hash.data(), hello.tokenizer_hash.size());
    hello.split_layer = reader.u32();
    hello.total_layers = reader.u32();
    hello.hidden_dim = reader.u32();
    hello.vocab_size = reader.u32();
    hello.max_tokens = reader.u32();
    hello.capabilities = reader.u32();
    return hello;
}

void read_exact(int fd, uint8_t * destination, size_t count) {
    size_t done = 0;
    while (done < count) {
        const ssize_t received = ::recv(fd, destination + done, count - done, 0);
        if (received == 0) throw ProtocolError("truncated split wire frame");
        if (received < 0) {
            if (errno == EINTR) continue;
            throw ProtocolError(std::string("split wire recv: ") + std::strerror(errno));
        }
        done += size_t(received);
    }
}

} // namespace

std::vector<uint8_t> encode_frame(const Frame & frame) {
    validate_frame_fields(frame);
    std::vector<uint8_t> payload;
    put64(payload, frame.request_id);
    put64(payload, frame.session_id);
    if (frame.type == Type::Hello || frame.type == Type::HelloAck) {
        put_handshake(payload, frame.handshake);
    } else if (frame.type == Type::Activation) {
        if (frame.activation_f16.size() > (kMaxFrameBytes - 128) / 2) throw ProtocolError("activation exceeds frame limit");
        put64(payload, frame.frame_id);
        put64(payload, frame.token_pos);
        put32(payload, frame.token_count);
        put64(payload, frame.candidate_id);
        put32(payload, frame.token_id);
        put64(payload, frame.predecessor_frame_id);
        put64(payload, frame.predecessor_pos);
        put32(payload, frame.predecessor_target_token);
        put32(payload, uint32_t(frame.activation_f16.size()));
        for (uint16_t value : frame.activation_f16) put16(payload, value);
    } else if (frame.type == Type::Reply) {
        put64(payload, frame.frame_id);
        put64(payload, frame.token_pos);
        put64(payload, frame.candidate_id);
        put8(payload, uint8_t(frame.decision));
        put32(payload, frame.target_token);
        put32(payload, frame.predecessor_target_token);
    }
    if (payload.size() > kMaxFrameBytes) throw ProtocolError("split wire frame exceeds limit");
    std::vector<uint8_t> bytes;
    bytes.reserve(kHeaderBytes + payload.size());
    put32(bytes, kMagic);
    put16(bytes, kVersion);
    put16(bytes, uint16_t(frame.type));
    put32(bytes, uint32_t(payload.size()));
    put32(bytes, crc32(payload.data(), payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

Frame decode_frame(const std::vector<uint8_t> & bytes) {
    if (bytes.size() < kHeaderBytes) throw ProtocolError("truncated split wire header");
    Reader header(bytes.data(), kHeaderBytes);
    if (header.u32() != kMagic) throw ProtocolError("bad split wire magic");
    if (header.u16() != kVersion) throw ProtocolError("unsupported split wire version");
    const Type type = Type(header.u16());
    validate_type(type);
    const uint32_t size = header.u32();
    const uint32_t checksum = header.u32();
    if (size > kMaxFrameBytes || bytes.size() - kHeaderBytes != size) throw ProtocolError("invalid split wire frame length");
    const uint8_t * payload = bytes.data() + kHeaderBytes;
    if (crc32(payload, size) != checksum) throw ProtocolError("split wire checksum mismatch");
    Reader reader(payload, size);
    Frame frame;
    frame.type = type;
    frame.request_id = reader.u64();
    frame.session_id = reader.u64();
    if (type == Type::Hello || type == Type::HelloAck) {
        frame.handshake = get_handshake(reader);
    } else if (type == Type::Activation) {
        frame.frame_id = reader.u64();
        frame.token_pos = reader.u64();
        frame.token_count = reader.u32();
        frame.candidate_id = reader.u64();
        frame.token_id = reader.u32();
        frame.predecessor_frame_id = reader.u64();
        frame.predecessor_pos = reader.u64();
        frame.predecessor_target_token = reader.u32();
        const uint32_t samples = reader.u32();
        if (samples > reader.remaining() / 2 || reader.remaining() != uint64_t(samples) * 2) throw ProtocolError("invalid FP16 activation length");
        frame.activation_f16.reserve(samples);
        for (uint32_t i = 0; i < samples; ++i) frame.activation_f16.push_back(reader.u16());
    } else if (type == Type::Reply) {
        frame.frame_id = reader.u64();
        frame.token_pos = reader.u64();
        frame.candidate_id = reader.u64();
        frame.decision = Decision(reader.u8());
        frame.target_token = reader.u32();
        frame.predecessor_target_token = reader.u32();
    }
    if (reader.remaining() != 0) throw ProtocolError("trailing split wire bytes");
    validate_frame_fields(frame);
    return frame;
}

void send_frame(int fd, const Frame & frame) {
    const std::vector<uint8_t> bytes = encode_frame(frame);
    size_t done = 0;
    while (done < bytes.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const ssize_t sent = ::send(fd, bytes.data() + done, bytes.size() - done, flags);
        if (sent < 0) {
            if (errno == EINTR) continue;
            throw ProtocolError(std::string("split wire send: ") + std::strerror(errno));
        }
        if (sent == 0) throw ProtocolError("split wire send returned zero");
        done += size_t(sent);
    }
}

Frame recv_frame(int fd) {
    std::vector<uint8_t> bytes(kHeaderBytes);
    read_exact(fd, bytes.data(), bytes.size());
    Reader header(bytes.data(), bytes.size());
    if (header.u32() != kMagic) throw ProtocolError("bad split wire magic");
    if (header.u16() != kVersion) throw ProtocolError("unsupported split wire version");
    validate_type(Type(header.u16()));
    const uint32_t size = header.u32();
    if (size > kMaxFrameBytes) throw ProtocolError("split wire frame exceeds limit");
    bytes.resize(kHeaderBytes + size);
    read_exact(fd, bytes.data() + kHeaderBytes, size);
    return decode_frame(bytes);
}

void validate_handshake(const Handshake & expected, const Handshake & actual) {
    if (expected.split_layer == 0 || expected.total_layers <= expected.split_layer || expected.hidden_dim == 0 || expected.vocab_size == 0 || expected.max_tokens == 0) {
        throw ProtocolError("invalid local split handshake");
    }
    if (expected.model_hash != actual.model_hash || expected.tokenizer_hash != actual.tokenizer_hash || expected.split_layer != actual.split_layer || expected.total_layers != actual.total_layers || expected.hidden_dim != actual.hidden_dim || expected.vocab_size != actual.vocab_size || expected.max_tokens != actual.max_tokens) {
        throw ProtocolError("split model or shape handshake mismatch");
    }
    if ((actual.capabilities & expected.capabilities) != expected.capabilities) throw ProtocolError("missing split protocol capability");
}

void validate_reply(const Frame & activation, const Frame & reply, Decision expected_decision, uint32_t expected_predecessor_target) {
    validate_frame_fields(activation);
    validate_frame_fields(reply);
    if (activation.type != Type::Activation || reply.type != Type::Reply || reply.request_id != activation.request_id || reply.session_id != activation.session_id || reply.frame_id != activation.frame_id || reply.token_pos != activation.token_pos || reply.candidate_id != activation.candidate_id || reply.decision != expected_decision) {
        throw ProtocolError("split reply identity or decision mismatch");
    }
    if (activation.candidate_id != kNoId) {
        if (expected_predecessor_target == kNoToken || reply.predecessor_target_token != expected_predecessor_target || (expected_decision == Decision::Rejected && reply.target_token != expected_predecessor_target)) {
            throw ProtocolError("early reply predecessor target mismatch");
        }
    }
}

WorkerState::WorkerState(Handshake expected) : expected_(expected) {
    validate_handshake(expected_, expected_);
}

void WorkerState::reset() {
    active_ = false;
    request_id_ = session_id_ = next_frame_id_ = next_token_pos_ = next_candidate_id_ = 0;
    pending_frame_id_ = pending_pos_ = pending_candidate_id_ = retained_frame_id_ = retained_pos_ = kNoId;
    pending_count_ = 0;
    retained_target_ = kNoToken;
}

void WorkerState::start(uint64_t request_id, uint64_t session_id) {
    if (request_id == 0 || session_id == 0) throw ProtocolError("zero request or session ID");
    reset();
    request_id_ = request_id;
    session_id_ = session_id;
    active_ = true;
}

Decision WorkerState::inspect_activation(const Frame & frame) {
    validate_frame_fields(frame);
    if (!active_ || frame.type != Type::Activation || frame.request_id != request_id_ || frame.session_id != session_id_) throw ProtocolError("activation session mismatch");
    if (pending_frame_id_ != kNoId) throw ProtocolError("activation before pending compute completed");
    if (frame.frame_id != next_frame_id_ || frame.token_pos != next_token_pos_) throw ProtocolError("activation frame or logical position out of order");
    if (frame.token_count > expected_.max_tokens - next_token_pos_ || frame.activation_f16.size() != uint64_t(frame.token_count) * expected_.hidden_dim) {
        throw ProtocolError("activation shape or position exceeds handshake");
    }
    if (frame.token_id != kNoToken && frame.token_id >= expected_.vocab_size) throw ProtocolError("activation input token outside vocabulary");
    if (frame.candidate_id != kNoId) {
        if ((expected_.capabilities & kEarlyLookahead) == 0 || frame.candidate_id != next_candidate_id_ || frame.token_id >= expected_.vocab_size || retained_frame_id_ == kNoId || frame.predecessor_frame_id != retained_frame_id_ || frame.predecessor_pos != retained_pos_ || frame.token_pos != retained_pos_ + 1) {
            throw ProtocolError("early candidate identity or token mismatch");
        }
        ++next_candidate_id_;
        ++next_frame_id_;
        if (frame.token_id != retained_target_) return Decision::Rejected;
        pending_frame_id_ = frame.frame_id;
        pending_pos_ = frame.token_pos;
        pending_count_ = frame.token_count;
        pending_candidate_id_ = frame.candidate_id;
        return Decision::Accepted;
    }
    ++next_frame_id_;
    pending_frame_id_ = frame.frame_id;
    pending_pos_ = frame.token_pos;
    pending_count_ = frame.token_count;
    pending_candidate_id_ = frame.candidate_id;
    return Decision::Ordinary;
}

void WorkerState::complete_activation(const Frame & frame, uint32_t target_token) {
    if (!active_ || pending_frame_id_ != frame.frame_id || pending_pos_ != frame.token_pos || pending_count_ != frame.token_count || pending_candidate_id_ != frame.candidate_id || frame.request_id != request_id_ || frame.session_id != session_id_ || target_token >= expected_.vocab_size) throw ProtocolError("invalid activation completion");
    next_token_pos_ += pending_count_;
    retained_frame_id_ = frame.frame_id;
    retained_pos_ = next_token_pos_ - 1;
    retained_target_ = target_token;
    pending_frame_id_ = kNoId;
    pending_pos_ = pending_candidate_id_ = kNoId;
    pending_count_ = 0;
}

} // namespace split_wire
