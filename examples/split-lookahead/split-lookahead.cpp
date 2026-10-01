#include "split-wire.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace {

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct Options {
    std::string mode;
    std::string model_path;
    std::string draft_path;
    std::string bind_address;
    std::string connect_address;
    std::string prompt_ids;
    std::string prompt_ids_file;
    std::string output_path;
    std::string send_policy = "off";
    uint32_t split = 0;
    uint32_t ctx = 0;
    uint32_t outputs = 0;
    uint32_t prefill_chunk = 256;
    uint32_t timeout_seconds = 600;
    int gpu_layers = 99;
    int threads = 2;
};

class Socket {
public:
    explicit Socket(int fd = -1) : fd_(fd) {}
    ~Socket() { if (fd_ >= 0) { close(fd_); } }
    Socket(const Socket &) = delete;
    Socket & operator=(const Socket &) = delete;
    Socket(Socket && other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Socket & operator=(Socket && other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) { close(fd_); }
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    int get() const { return fd_; }
private:
    int fd_;
};

uint32_t parse_u32(const std::string & value, const char * name) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return result;
}

int parse_int(const std::string & value, const char * name) {
    const uint32_t parsed = parse_u32(value, name);
    if (parsed > INT32_MAX) { throw std::runtime_error(std::string(name) + " exceeds INT32_MAX"); }
    return static_cast<int>(parsed);
}

void print_usage(const char * executable) {
    std::fprintf(stderr, "usage: %s worker --model TARGET.gguf --split K --ctx N --bind HOST:PORT [--threads N] [--gpu-layers N]\n", executable);
    std::fprintf(stderr, "       %s coordinator --model TARGET.gguf --draft-model DRAFT.gguf --split K --ctx N --connect HOST:PORT --prompt-ids 1,2,3 --outputs N [--send-policy early|off] [--output FILE]\n", executable);
    std::fprintf(stderr, "       %s single --model TARGET.gguf --ctx N --prompt-ids-file INPUT.json --outputs N [--output FILE]\n", executable);
    std::fprintf(stderr, "       --prompt-ids-file accepts comma-separated IDs or a JSON integer array. --send-policy off does not require a draft model.\n");
    std::fprintf(stderr, "       Optional: --prefill-chunk N (256), --timeout-seconds N (600), --threads N (2), --gpu-layers N (99). Single needs no split, draft or network.\n");
}

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "worker" || key == "coordinator" || key == "single") {
            if (!options.mode.empty()) { throw std::runtime_error("mode specified more than once"); }
            options.mode = key;
            continue;
        }
        if (i + 1 == argc) { throw std::runtime_error("missing value for " + key); }
        std::string value = argv[++i];
        if (key == "--mode") { options.mode = value; }
        else if (key == "--model") { options.model_path = value; }
        else if (key == "--draft-model") { options.draft_path = value; }
        else if (key == "--bind") { options.bind_address = value; }
        else if (key == "--connect") { options.connect_address = value; }
        else if (key == "--prompt-ids") { options.prompt_ids = value; }
        else if (key == "--prompt-ids-file") { options.prompt_ids_file = value; }
        else if (key == "--output") { options.output_path = value; }
        else if (key == "--send-policy") { options.send_policy = value; }
        else if (key == "--split") { options.split = parse_u32(value, "split"); }
        else if (key == "--ctx") { options.ctx = parse_u32(value, "ctx"); }
        else if (key == "--outputs") { options.outputs = parse_u32(value, "outputs"); }
        else if (key == "--prefill-chunk") { options.prefill_chunk = parse_u32(value, "prefill-chunk"); }
        else if (key == "--timeout-seconds") { options.timeout_seconds = parse_u32(value, "timeout-seconds"); }
        else if (key == "--gpu-layers") { options.gpu_layers = parse_int(value, "gpu-layers"); }
        else if (key == "--threads") { options.threads = parse_int(value, "threads"); }
        else { throw std::runtime_error("unknown option: " + key); }
    }
    if (options.mode != "worker" && options.mode != "coordinator" && options.mode != "single") { throw std::runtime_error("mode must be worker, coordinator or single"); }
    if (options.model_path.empty() || options.ctx == 0 || options.ctx > (options.mode == "single" ? INT32_MAX : INT32_MAX / 2) || options.outputs > INT32_MAX) { throw std::runtime_error("--model and positive --ctx are required; --ctx exceeds the supported cache size"); }
    if (options.mode != "single" && options.split == 0) { throw std::runtime_error("worker and coordinator require positive --split"); }
    if (options.threads <= 0 || options.prefill_chunk == 0 || options.timeout_seconds == 0 || options.timeout_seconds > INT32_MAX) { throw std::runtime_error("--threads, --prefill-chunk and --timeout-seconds must be positive"); }
    if (options.mode == "worker") {
        if (options.bind_address.empty()) { throw std::runtime_error("worker requires --bind HOST:PORT"); }
    } else if (options.mode == "coordinator") {
        if (options.connect_address.empty() || options.outputs == 0) { throw std::runtime_error("coordinator requires --connect and --outputs"); }
        if (options.prompt_ids.empty() == options.prompt_ids_file.empty()) { throw std::runtime_error("provide exactly one of --prompt-ids or --prompt-ids-file"); }
        if (options.send_policy != "early" && options.send_policy != "off") { throw std::runtime_error("--send-policy must be early or off"); }
        if (options.send_policy == "early" && options.draft_path.empty()) { throw std::runtime_error("early policy requires --draft-model"); }
    } else {
        if (options.outputs == 0) { throw std::runtime_error("single requires --outputs"); }
        if (options.prompt_ids.empty() == options.prompt_ids_file.empty()) { throw std::runtime_error("provide exactly one of --prompt-ids or --prompt-ids-file"); }
        if (options.split != 0 || !options.draft_path.empty() || !options.bind_address.empty() || !options.connect_address.empty() || options.send_policy != "off") { throw std::runtime_error("single does not use --split, --draft-model, endpoints or --send-policy"); }
    }
    return options;
}

std::vector<llama_token> parse_prompt_ids(const Options & options) {
    std::string input = options.prompt_ids;
    if (!options.prompt_ids_file.empty()) {
        std::ifstream stream(options.prompt_ids_file);
        if (!stream) { throw std::runtime_error("could not open prompt IDs file"); }
        input.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    }
    std::vector<llama_token> ids;
    size_t pos = 0;
    while (pos < input.size()) {
        while (pos < input.size() && (input[pos] == ',' || input[pos] == '[' || input[pos] == ']' || input[pos] == ' ' || input[pos] == '\n' || input[pos] == '\r' || input[pos] == '\t')) { ++pos; }
        if (pos == input.size()) { break; }
        const char * start = input.data() + pos;
        uint32_t id = 0;
        auto parsed = std::from_chars(start, input.data() + input.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr == start || id > INT32_MAX) { throw std::runtime_error("invalid prompt token ID"); }
        pos = static_cast<size_t>(parsed.ptr - input.data());
        if (pos < input.size() && input[pos] != ',' && input[pos] != ']' && input[pos] != ' ' && input[pos] != '\n' && input[pos] != '\r' && input[pos] != '\t') { throw std::runtime_error("invalid prompt ID separator"); }
        ids.push_back(static_cast<llama_token>(id));
    }
    if (ids.empty()) { throw std::runtime_error("prompt must contain at least one token"); }
    return ids;
}

std::array<uint8_t, 32> hash_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("could not open GGUF for SHA256: " + path); }
    sha256_t state;
    sha256_init(&state);
    std::array<char, 1 << 20> buffer{};
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        const auto count = stream.gcount();
        if (count > 0) { sha256_update(&state, reinterpret_cast<const unsigned char *>(buffer.data()), static_cast<size_t>(count)); }
    }
    if (!stream.eof()) { throw std::runtime_error("could not finish GGUF SHA256: " + path); }
    std::array<uint8_t, 32> digest{};
    sha256_final(&state, digest.data());
    return digest;
}

void hash_u32(sha256_t & state, uint32_t value) {
    const unsigned char bytes[] = { static_cast<unsigned char>(value >> 24), static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 8), static_cast<unsigned char>(value) };
    sha256_update(&state, bytes, sizeof(bytes));
}

std::array<uint8_t, 32> hash_vocab(const llama_vocab * vocab, bool include_bos = true) {
    sha256_t state;
    sha256_init(&state);
    const uint32_t size = static_cast<uint32_t>(llama_vocab_n_tokens(vocab));
    hash_u32(state, size);
    for (uint32_t i = 0; i < size; ++i) {
        const char * text = llama_vocab_get_text(vocab, static_cast<llama_token>(i));
        const size_t len = text ? std::strlen(text) : 0;
        hash_u32(state, static_cast<uint32_t>(len));
        if (len) { sha256_update(&state, reinterpret_cast<const unsigned char *>(text), len); }
        hash_u32(state, static_cast<uint32_t>(llama_vocab_get_attr(vocab, static_cast<llama_token>(i))));
    }
    if (include_bos) { hash_u32(state, static_cast<uint32_t>(llama_vocab_bos(vocab))); }
    hash_u32(state, static_cast<uint32_t>(llama_vocab_eos(vocab)));
    hash_u32(state, static_cast<uint32_t>(llama_vocab_eot(vocab)));
    std::array<uint8_t, 32> digest{};
    sha256_final(&state, digest.data());
    return digest;
}

Model load_model(const std::string & path, int gpu_layers, uint32_t layer_start, uint32_t layer_end) {
    const gguf_init_params metadata_params = { true, nullptr };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(path.c_str(), metadata_params), gguf_free);
    if (!metadata) { throw std::runtime_error("failed to read GGUF metadata: " + path); }
    const int64_t split_count = gguf_find_key(metadata.get(), "split.count");
    if (split_count >= 0 && (gguf_get_kv_type(metadata.get(), split_count) != GGUF_TYPE_UINT16 || gguf_get_val_u16(metadata.get(), split_count) != 1)) {
        throw std::runtime_error("split-lookahead requires a single-file GGUF for complete model identity checking");
    }
    metadata.reset();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    params.stage_layer_start = layer_start;
    params.stage_layer_end = layer_end;
    Model model(llama_model_load_from_file(path.c_str(), params), llama_model_free);
    if (!model) { throw std::runtime_error("failed to load GGUF: " + path); }
    return model;
}

Context make_context(llama_model * model, uint32_t ctx_size, bool embeddings, uint32_t n_seq_max, int threads, uint32_t prefill_chunk) {
    llama_context_params params = llama_context_default_params();
    if (ctx_size > UINT32_MAX / n_seq_max) { throw std::runtime_error("context size overflow"); }
    params.n_ctx = ctx_size * n_seq_max;
    params.n_batch = std::max(std::min(ctx_size, prefill_chunk), n_seq_max);
    params.n_ubatch = std::max(std::min(params.n_batch, 512U), n_seq_max);
    params.n_seq_max = n_seq_max;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.kv_unified = true;
    params.embeddings = embeddings;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    Context ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx) { throw std::runtime_error("failed to create model context"); }
    return ctx;
}

llama_token greedy(llama_context * ctx, uint32_t vocab_size) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) { throw std::runtime_error("model did not return logits"); }
    uint32_t best = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(logits[i])) { throw std::runtime_error("model produced nonfinite logits"); }
        if (logits[i] > maximum) { maximum = logits[i]; best = i; }
    }
    return static_cast<llama_token>(best);
}

class PrefixSession {
public:
    PrefixSession(llama_model * model, uint32_t ctx_size, int threads, uint32_t prefill_chunk) : ctx_(make_context(model, ctx_size, true, 2, threads, prefill_chunk)), width_(static_cast<uint32_t>(llama_model_n_embd(model))) {}

    std::vector<uint16_t> forward(const std::vector<llama_token> & tokens, uint32_t pos, llama_seq_id seq) {
        if (tokens.empty()) { throw std::runtime_error("empty prefix batch"); }
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = static_cast<llama_pos>(pos + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = 1;
        }
        const int status = llama_decode(ctx_.get(), batch);
        llama_batch_free(batch);
        if (status != 0) { throw std::runtime_error("prefix decode failed"); }
        std::vector<uint16_t> result(tokens.size() * width_);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const float * row = llama_get_embeddings_ith(ctx_.get(), static_cast<int32_t>(i));
            if (!row) { throw std::runtime_error("prefix activation missing"); }
            for (uint32_t j = 0; j < width_; ++j) {
                if (!std::isfinite(row[j])) { throw std::runtime_error("prefix produced nonfinite activation"); }
                result[i * width_ + j] = ggml_fp32_to_fp16(row[j]);
                if (!std::isfinite(ggml_fp16_to_fp32(result[i * width_ + j]))) { throw std::runtime_error("prefix activation overflows FP16"); }
            }
        }
        return result;
    }

    void begin() {
        if (branch_active_) { throw std::runtime_error("prefix branch already active"); }
        llama_memory_seq_cp(llama_get_memory(ctx_.get()), 0, 1, -1, -1);
        branch_active_ = true;
    }

    void commit() {
        if (!branch_active_) { throw std::runtime_error("no prefix branch to commit"); }
        llama_memory_t memory = llama_get_memory(ctx_.get());
        if (!llama_memory_seq_rm(memory, 0, -1, -1)) { throw std::runtime_error("could not clear base prefix cache"); }
        llama_memory_seq_cp(memory, 1, 0, -1, -1);
        if (!llama_memory_seq_rm(memory, 1, -1, -1)) { throw std::runtime_error("could not clear committed prefix branch"); }
        branch_active_ = false;
    }

    void discard() {
        if (!branch_active_) { throw std::runtime_error("no prefix branch to discard"); }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_.get()), 1, -1, -1)) { throw std::runtime_error("could not discard prefix branch"); }
        branch_active_ = false;
    }

    void reset() { llama_memory_clear(llama_get_memory(ctx_.get()), true); branch_active_ = false; }

private:
    Context ctx_;
    uint32_t width_;
    bool branch_active_ = false;
};

class SuffixSession {
public:
    SuffixSession(llama_model * model, uint32_t ctx_size, int threads, uint32_t prefill_chunk) : ctx_(make_context(model, ctx_size, false, 1, threads, prefill_chunk)), width_(static_cast<uint32_t>(llama_model_n_embd(model))), vocab_size_(static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)))) {}

    llama_token forward(const std::vector<uint16_t> & activation, uint32_t pos, uint32_t count) {
        if (count == 0 || activation.size() != static_cast<size_t>(count) * width_) { throw std::runtime_error("suffix activation size mismatch"); }
        std::vector<float> embeddings(static_cast<size_t>(count) * width_);
        std::vector<llama_pos> positions(static_cast<size_t>(count) * 4);
        std::vector<int32_t> n_seq_id(count, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(count, &seq);
        std::vector<int8_t> logits(count, 0);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < width_; ++j) {
                embeddings[i * width_ + j] = ggml_fp16_to_fp32(activation[i * width_ + j]);
                if (!std::isfinite(embeddings[i * width_ + j])) { throw std::runtime_error("nonfinite FP16 boundary input"); }
            }
            positions[i] = positions[count + i] = positions[2 * count + i] = static_cast<llama_pos>(pos + i);
            positions[3 * count + i] = 0;
            logits[i] = i + 1 == count;
        }
        llama_batch batch{};
        batch.n_tokens = static_cast<int32_t>(count);
        batch.embd = embeddings.data();
        batch.pos = positions.data();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id = seq_ids.data();
        batch.logits = logits.data();
        const int status = llama_decode(ctx_.get(), batch);
        if (status != 0) { throw std::runtime_error("suffix decode failed"); }
        return greedy(ctx_.get(), vocab_size_);
    }

    void reset() { llama_memory_clear(llama_get_memory(ctx_.get()), true); }

private:
    Context ctx_;
    uint32_t width_;
    uint32_t vocab_size_;
};

class DraftSession {
public:
    DraftSession(llama_model * model, uint32_t ctx_size, int threads, uint32_t prefill_chunk) :
        ctx_(make_context(model, ctx_size, false, 1, threads, prefill_chunk)),
        vocab_size_(static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)))),
        prefill_chunk_(prefill_chunk) {}

    llama_token propose(const std::vector<llama_token> & confirmed) {
        if (confirmed.size() <= processed_) { throw std::runtime_error("draft has no new confirmed tokens"); }
        while (processed_ < confirmed.size()) {
            const size_t count = std::min<size_t>(prefill_chunk_, confirmed.size() - processed_);
            llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
            batch.n_tokens = static_cast<int32_t>(count);
            for (size_t i = 0; i < count; ++i) {
                batch.token[i] = confirmed[processed_ + i];
                batch.pos[i] = static_cast<llama_pos>(processed_ + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = processed_ + i + 1 == confirmed.size();
            }
            const int status = llama_decode(ctx_.get(), batch);
            llama_batch_free(batch);
            if (status != 0) { throw std::runtime_error("draft decode failed"); }
            processed_ += count;
        }
        return greedy(ctx_.get(), vocab_size_);
    }

    void reset() { llama_memory_clear(llama_get_memory(ctx_.get()), true); processed_ = 0; }

private:
    Context ctx_;
    uint32_t vocab_size_;
    uint32_t prefill_chunk_;
    size_t processed_ = 0;
};

class SingleSession {
public:
    SingleSession(llama_model * model, uint32_t ctx_size, int threads, uint32_t prefill_chunk) :
        ctx_(make_context(model, ctx_size, false, 1, threads, prefill_chunk)),
        vocab_size_(static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)))) {}

    llama_token forward(const llama_token * tokens, uint32_t count, uint32_t pos) {
        if (!tokens || count == 0) { throw std::runtime_error("empty single-model batch"); }
        llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
        batch.n_tokens = static_cast<int32_t>(count);
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = static_cast<llama_pos>(pos + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == count;
        }
        const int status = llama_decode(ctx_.get(), batch);
        llama_batch_free(batch);
        if (status != 0) { throw std::runtime_error("single-model decode failed"); }
        return greedy(ctx_.get(), vocab_size_);
    }

private:
    Context ctx_;
    uint32_t vocab_size_;
};

std::pair<std::string, std::string> split_host_port(const std::string & address) {
    if (address.empty()) { throw std::runtime_error("empty network address"); }
    std::string host;
    std::string port;
    if (address.front() == '[') {
        const size_t end = address.find(']');
        if (end == std::string::npos || end + 1 >= address.size() || address[end + 1] != ':') { throw std::runtime_error("invalid IPv6 address"); }
        host = address.substr(1, end - 1);
        port = address.substr(end + 2);
    } else {
        const size_t colon = address.rfind(':');
        if (colon == std::string::npos) { throw std::runtime_error("expected HOST:PORT"); }
        host = address.substr(0, colon);
        port = address.substr(colon + 1);
    }
    if (port.empty()) { throw std::runtime_error("empty network port"); }
    return {host, port};
}

Socket open_socket(const std::string & address, bool listening) {
    const auto [host, port] = split_host_port(address);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = listening ? AI_PASSIVE : 0;
    addrinfo * addresses = nullptr;
    const int status = getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &addresses);
    if (status != 0) { throw std::runtime_error(std::string("getaddrinfo failed: ") + gai_strerror(status)); }
    Socket socket;
    for (addrinfo * current = addresses; current; current = current->ai_next) {
        Socket attempt(::socket(current->ai_family, current->ai_socktype, current->ai_protocol));
        if (attempt.get() < 0) { continue; }
        if (listening) {
            const int yes = 1;
            setsockopt(attempt.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
            if (::bind(attempt.get(), current->ai_addr, current->ai_addrlen) == 0 && ::listen(attempt.get(), 1) == 0) { socket = std::move(attempt); break; }
        } else if (::connect(attempt.get(), current->ai_addr, current->ai_addrlen) == 0) {
            socket = std::move(attempt);
            break;
        }
    }
    freeaddrinfo(addresses);
    if (socket.get() < 0) { throw std::runtime_error(listening ? "could not bind/listen" : "could not connect"); }
    return socket;
}

void configure_peer_socket(int fd, uint32_t timeout_seconds) {
    const int yes = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) != 0) { throw std::runtime_error("could not set TCP_NODELAY"); }
    timeval timeout{};
    timeout.tv_sec = static_cast<time_t>(timeout_seconds);
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 || setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) { throw std::runtime_error("could not set socket timeout"); }
}

split_wire::Handshake handshake_for(const Options & options, llama_model * model) {
    split_wire::Handshake h;
    h.model_hash = hash_file(options.model_path);
    h.tokenizer_hash = hash_vocab(llama_model_get_vocab(model));
    h.split_layer = options.split;
    h.total_layers = static_cast<uint32_t>(llama_model_n_layer(model));
    h.hidden_dim = static_cast<uint32_t>(llama_model_n_embd(model));
    h.vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    h.max_tokens = options.ctx;
    h.capabilities = split_wire::kEarlyLookahead;
    return h;
}

void validate_vocab(const llama_vocab * target, const llama_vocab * draft) {
    const bool target_add_bos = llama_vocab_get_add_bos(target);
    const bool draft_add_bos = llama_vocab_get_add_bos(draft);
    // Prepared IDs never insert BOS. Some compatible GGUF conversions omit its
    // metadata; ignore that unused ID only when both tokenizers disable insertion.
    if (target_add_bos != draft_add_bos || hash_vocab(target, target_add_bos) != hash_vocab(draft, draft_add_bos)) {
        throw std::runtime_error("target and draft tokenizer maps differ");
    }
}

void check_reply(const split_wire::Frame & reply, const split_wire::Frame & request, split_wire::Decision decision, uint32_t vocab_size, uint32_t expected_predecessor = UINT32_MAX) {
    split_wire::validate_reply(request, reply, decision, expected_predecessor);
    if (reply.target_token >= vocab_size) { throw std::runtime_error("worker returned invalid target token"); }
}

split_wire::Frame activation_frame(uint64_t request_id, uint64_t session_id, uint64_t frame_id, uint32_t pos, uint32_t count, uint32_t token_id, std::vector<uint16_t> && activation) {
    split_wire::Frame frame;
    frame.type = split_wire::Type::Activation;
    frame.request_id = request_id;
    frame.session_id = session_id;
    frame.frame_id = frame_id;
    frame.token_pos = pos;
    frame.token_count = count;
    frame.token_id = token_id;
    frame.activation_f16 = std::move(activation);
    return frame;
}

void write_result(const Options & options, uint32_t prompt_tokens, const std::vector<llama_token> & output, uint32_t proposals, uint32_t accepted, uint32_t rejected, uint32_t worker_decode_calls, uint32_t wire_frames, double ttft_ms, double total_ms, double decode_tokens_per_second) {
    std::string json = "{\"runtime\":\"llama.cpp-split-lookahead\",\"timing_semantics\":\"post-handshake-prepared-input-to-last-output-v1\",\"schedule\":\"" + std::string(options.send_policy == "early" ? "rolling-early" : "off") + "\",\"speculative_schedule\":\"" + std::string(options.send_policy == "early" ? "rolling-early" : "none") + "\",\"tokens\":[";
    for (size_t i = 0; i < output.size(); ++i) {
        if (i) { json += ','; }
        json += std::to_string(output[i]);
    }
    json += "],\"output_tokens\":" + std::to_string(output.size());
    json += ",\"proposals\":" + std::to_string(proposals);
    json += ",\"speculative_emitted_tokens\":" + std::to_string(proposals);
    json += ",\"accepted\":" + std::to_string(accepted);
    json += ",\"rejected\":" + std::to_string(rejected);
    json += ",\"worker_decode_calls\":" + std::to_string(worker_decode_calls);
    json += ",\"wire_activation_frames\":" + std::to_string(wire_frames);
    json += ",\"codec\":\"fp16\",\"split\":" + std::to_string(options.split);
    json += ",\"prompt_tokens\":" + std::to_string(prompt_tokens);
    json += ",\"ttft_ms\":" + std::to_string(ttft_ms);
    json += ",\"total_ms\":" + std::to_string(total_ms);
    json += ",\"decode_tokens_per_second\":" + (output.size() > 1 ? std::to_string(decode_tokens_per_second) : "null") + "}\n";
    if (options.output_path.empty()) {
        std::fwrite(json.data(), 1, json.size(), stdout);
    } else {
        std::ofstream stream(options.output_path, std::ios::binary);
        if (!stream) { throw std::runtime_error("could not open output JSON file"); }
        stream << json;
        if (!stream) { throw std::runtime_error("could not write output JSON file"); }
    }
}

void write_single_result(const Options & options, uint32_t prompt_tokens, const std::vector<llama_token> & output, double ttft_ms, double total_ms, double decode_tokens_per_second) {
    std::string json = "{\"runtime\":\"llama.cpp-single\",\"mode\":\"single\",\"timing_semantics\":\"post-context-prepared-input-to-last-output-v1\",\"tokens\":[";
    for (size_t i = 0; i < output.size(); ++i) {
        if (i) { json += ','; }
        json += std::to_string(output[i]);
    }
    json += "],\"output_tokens\":" + std::to_string(output.size());
    json += ",\"prompt_tokens\":" + std::to_string(prompt_tokens);
    json += ",\"target_decode_calls\":" + std::to_string(output.size() - 1);
    json += ",\"ttft_ms\":" + std::to_string(ttft_ms);
    json += ",\"total_ms\":" + std::to_string(total_ms);
    json += ",\"decode_tokens_per_second\":" + (output.size() > 1 ? std::to_string(decode_tokens_per_second) : "null") + "}\n";
    if (options.output_path.empty()) {
        std::fwrite(json.data(), 1, json.size(), stdout);
    } else {
        std::ofstream stream(options.output_path, std::ios::binary);
        if (!stream) { throw std::runtime_error("could not open output JSON file"); }
        stream << json;
        if (!stream) { throw std::runtime_error("could not write output JSON file"); }
    }
}

void run_single(const Options & options, llama_model * model) {
    const std::vector<llama_token> prompt = parse_prompt_ids(options);
    if (prompt.size() > options.ctx || static_cast<uint64_t>(prompt.size()) + options.outputs - 1 > options.ctx) { throw std::runtime_error("prompt plus output budget exceeds --ctx"); }
    const uint32_t vocab_size = static_cast<uint32_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    for (llama_token token : prompt) {
        if (token < 0 || static_cast<uint32_t>(token) >= vocab_size) { throw std::runtime_error("prompt token is out of vocabulary"); }
    }
    SingleSession target(model, options.ctx, options.threads, options.prefill_chunk);
    std::vector<llama_token> output;
    output.reserve(options.outputs);
    using Clock = std::chrono::steady_clock;
    const auto request_started = Clock::now();
    llama_token first_token = LLAMA_TOKEN_NULL;
    for (size_t offset = 0; offset < prompt.size(); offset += options.prefill_chunk) {
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(options.prefill_chunk, prompt.size() - offset));
        first_token = target.forward(prompt.data() + offset, count, static_cast<uint32_t>(offset));
    }
    output.push_back(first_token);
    const auto first_token_at = Clock::now();
    while (output.size() < options.outputs) {
        const uint32_t pos = static_cast<uint32_t>(prompt.size() + output.size() - 1);
        const llama_token current = output.back();
        output.push_back(target.forward(&current, 1, pos));
    }
    const auto finished_at = Clock::now();
    const double ttft_ms = std::chrono::duration<double, std::milli>(first_token_at - request_started).count();
    const double total_ms = std::chrono::duration<double, std::milli>(finished_at - request_started).count();
    const double decode_seconds = std::chrono::duration<double>(finished_at - first_token_at).count();
    const double decode_tokens_per_second = output.size() > 1 && decode_seconds > 0 ? static_cast<double>(output.size() - 1) / decode_seconds : 0.0;
    write_single_result(options, static_cast<uint32_t>(prompt.size()), output, ttft_ms, total_ms, decode_tokens_per_second);
}

void run_worker(const Options & options, llama_model * model, const split_wire::Handshake & local) {
    SuffixSession suffix(model, options.ctx, options.threads, options.prefill_chunk);
    Socket listener = open_socket(options.bind_address, true);
    std::fprintf(stderr, "split-lookahead worker listening on %s\n", options.bind_address.c_str());
    Socket peer(::accept(listener.get(), nullptr, nullptr));
    if (peer.get() < 0) { throw std::runtime_error("accept failed"); }
    configure_peer_socket(peer.get(), options.timeout_seconds);
    const split_wire::Frame hello = split_wire::recv_frame(peer.get());
    if (hello.type != split_wire::Type::Hello) { throw std::runtime_error("expected handshake"); }
    split_wire::validate_handshake(local, hello.handshake);
    split_wire::Frame hello_ack;
    hello_ack.type = split_wire::Type::HelloAck;
    hello_ack.request_id = hello.request_id;
    hello_ack.session_id = hello.session_id;
    hello_ack.handshake = local;
    split_wire::send_frame(peer.get(), hello_ack);
    split_wire::WorkerState state(local);
    bool started = false;
    for (;;) {
        const split_wire::Frame frame = split_wire::recv_frame(peer.get());
        if (frame.type == split_wire::Type::Close) {
            if (!started || frame.request_id != hello.request_id || frame.session_id != hello.session_id) { throw std::runtime_error("close identity mismatch"); }
            suffix.reset();
            state.reset();
            return;
        }
        if (frame.type != split_wire::Type::Activation) { throw std::runtime_error("expected activation"); }
        if (!started) {
            if (frame.frame_id != 0 || frame.token_pos != 0) { throw std::runtime_error("request must start at zero"); }
            if (frame.request_id != hello.request_id || frame.session_id != hello.session_id) { throw std::runtime_error("activation handshake identity mismatch"); }
            suffix.reset();
            state.start(frame.request_id, frame.session_id);
            started = true;
        }
        const split_wire::Decision decision = state.inspect_activation(frame);
        split_wire::Frame reply;
        reply.type = split_wire::Type::Reply;
        reply.request_id = frame.request_id;
        reply.session_id = frame.session_id;
        reply.frame_id = frame.frame_id;
        reply.token_pos = frame.token_pos;
        reply.candidate_id = frame.candidate_id;
        reply.decision = decision;
        const uint32_t predecessor_target = frame.candidate_id == split_wire::kNoId ? split_wire::kNoToken : state.retained_target_token();
        if (decision == split_wire::Decision::Rejected) {
            reply.predecessor_target_token = predecessor_target;
            reply.target_token = reply.predecessor_target_token;
        } else {
            const llama_token target = suffix.forward(frame.activation_f16, static_cast<uint32_t>(frame.token_pos), frame.token_count);
            if (target < 0 || static_cast<uint32_t>(target) >= local.vocab_size) { throw std::runtime_error("invalid worker target token"); }
            state.complete_activation(frame, static_cast<uint32_t>(target));
            reply.target_token = static_cast<uint32_t>(target);
            if (decision == split_wire::Decision::Accepted) { reply.predecessor_target_token = predecessor_target; }
        }
        split_wire::send_frame(peer.get(), reply);
    }
}

void run_coordinator(const Options & options, llama_model * model, const split_wire::Handshake & local) {
    std::vector<llama_token> prompt = parse_prompt_ids(options);
    if (prompt.size() > options.ctx || static_cast<uint64_t>(prompt.size()) + options.outputs - 1 > options.ctx) { throw std::runtime_error("prompt plus output budget exceeds --ctx"); }
    for (llama_token token : prompt) {
        if (token < 0 || static_cast<uint32_t>(token) >= local.vocab_size) { throw std::runtime_error("prompt token is out of vocabulary"); }
    }
    PrefixSession prefix(model, options.ctx, options.threads, options.prefill_chunk);
    Model draft_model(nullptr, llama_model_free);
    std::unique_ptr<DraftSession> draft;
    if (options.send_policy == "early") {
        draft_model = load_model(options.draft_path, options.gpu_layers, 0, 0);
        validate_vocab(llama_model_get_vocab(model), llama_model_get_vocab(draft_model.get()));
        draft = std::make_unique<DraftSession>(draft_model.get(), options.ctx, options.threads, options.prefill_chunk);
    }
    Socket peer = open_socket(options.connect_address, false);
    configure_peer_socket(peer.get(), options.timeout_seconds);
    const uint64_t session_id = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^ static_cast<uint64_t>(getpid());
    const uint64_t request_id = 1;
    split_wire::Frame hello;
    hello.type = split_wire::Type::Hello;
    hello.request_id = request_id;
    hello.session_id = session_id ? session_id : 1;
    hello.handshake = local;
    split_wire::send_frame(peer.get(), hello);
    const split_wire::Frame hello_ack = split_wire::recv_frame(peer.get());
    if (hello_ack.type != split_wire::Type::HelloAck || hello_ack.request_id != hello.request_id || hello_ack.session_id != hello.session_id) { throw std::runtime_error("worker did not acknowledge handshake"); }
    split_wire::validate_handshake(local, hello_ack.handshake);

    uint64_t next_frame_id = 0;
    uint32_t wire_frames = 0;
    uint32_t worker_decode_calls = 0;
    uint32_t proposals = 0;
    uint32_t accepted = 0;
    uint32_t rejected = 0;
    std::vector<llama_token> output;
    std::vector<llama_token> confirmed = prompt;
    using Clock = std::chrono::steady_clock;
    const auto request_started = Clock::now();
    llama_token first_token = LLAMA_TOKEN_NULL;
    for (size_t offset = 0; offset < prompt.size(); offset += options.prefill_chunk) {
        const size_t count = std::min<size_t>(options.prefill_chunk, prompt.size() - offset);
        std::vector<llama_token> chunk(prompt.begin() + offset, prompt.begin() + offset + count);
        auto prompt_activation = prefix.forward(chunk, static_cast<uint32_t>(offset), 0);
        auto prefill = activation_frame(hello.request_id, hello.session_id, next_frame_id++, static_cast<uint32_t>(offset), static_cast<uint32_t>(count), UINT32_MAX, std::move(prompt_activation));
        split_wire::send_frame(peer.get(), prefill);
        ++wire_frames;
        auto prefill_reply = split_wire::recv_frame(peer.get());
        check_reply(prefill_reply, prefill, split_wire::Decision::Ordinary, local.vocab_size);
        first_token = static_cast<llama_token>(prefill_reply.target_token);
    }
    output.push_back(first_token);
    confirmed.push_back(output.back());
    const auto first_token_at = Clock::now();

    struct Pending {
        split_wire::Frame frame;
        uint32_t predecessor_target = UINT32_MAX;
    };
    std::unique_ptr<Pending> pending;
    while (output.size() < options.outputs) {
        const uint32_t pos = static_cast<uint32_t>(prompt.size() + output.size() - 1);
        if (!pending) {
            const llama_token current = output.back();
            auto activation = prefix.forward({current}, pos, 0);
            auto frame = activation_frame(hello.request_id, hello.session_id, next_frame_id++, pos, 1, static_cast<uint32_t>(current), std::move(activation));
            split_wire::send_frame(peer.get(), frame);
            ++wire_frames;
            pending = std::make_unique<Pending>(Pending{std::move(frame), UINT32_MAX});
        }
        if (options.send_policy == "off" || output.size() + 1 == options.outputs) {
            auto reply = split_wire::recv_frame(peer.get());
            const auto expected = pending->frame.candidate_id == UINT64_MAX ? split_wire::Decision::Ordinary : split_wire::Decision::Accepted;
            check_reply(reply, pending->frame, expected, local.vocab_size, pending->predecessor_target);
            output.push_back(static_cast<llama_token>(reply.target_token));
            confirmed.push_back(output.back());
            ++worker_decode_calls;
            pending.reset();
            continue;
        }

        const llama_token candidate = draft->propose(confirmed);
        if (candidate < 0 || static_cast<uint32_t>(candidate) >= local.vocab_size) { throw std::runtime_error("draft token is out of target vocabulary"); }
        prefix.begin();
        auto candidate_activation = prefix.forward({candidate}, pos + 1, 1);
        auto proposal = activation_frame(hello.request_id, hello.session_id, next_frame_id++, pos + 1, 1, static_cast<uint32_t>(candidate), std::move(candidate_activation));
        proposal.candidate_id = proposals;
        proposal.predecessor_frame_id = pending->frame.frame_id;
        proposal.predecessor_pos = pending->frame.token_pos;
        split_wire::send_frame(peer.get(), proposal);
        ++wire_frames;
        ++proposals;

        auto predecessor_reply = split_wire::recv_frame(peer.get());
        const auto expected = pending->frame.candidate_id == UINT64_MAX ? split_wire::Decision::Ordinary : split_wire::Decision::Accepted;
        check_reply(predecessor_reply, pending->frame, expected, local.vocab_size, pending->predecessor_target);
        ++worker_decode_calls;
        const uint32_t target = predecessor_reply.target_token;
        output.push_back(static_cast<llama_token>(target));
        confirmed.push_back(output.back());
        pending.reset();
        if (static_cast<uint32_t>(candidate) == target) {
            prefix.commit();
            ++accepted;
            pending = std::make_unique<Pending>(Pending{std::move(proposal), target});
        } else {
            prefix.discard();
            ++rejected;
            auto rejection_reply = split_wire::recv_frame(peer.get());
            check_reply(rejection_reply, proposal, split_wire::Decision::Rejected, local.vocab_size, target);
        }
    }
    if (options.send_policy == "early" && (proposals != options.outputs - (options.outputs > 1 ? 2 : 1) || accepted + rejected != proposals)) { throw std::runtime_error("rolling proposal accounting mismatch"); }
    if (worker_decode_calls != options.outputs - 1) { throw std::runtime_error("worker decode accounting mismatch"); }
    const auto finished_at = Clock::now();
    split_wire::Frame close_frame;
    close_frame.type = split_wire::Type::Close;
    close_frame.request_id = hello.request_id;
    close_frame.session_id = hello.session_id;
    split_wire::send_frame(peer.get(), close_frame);
    prefix.reset();
    if (draft) { draft->reset(); }
    const double ttft_ms = std::chrono::duration<double, std::milli>(first_token_at - request_started).count();
    const double total_ms = std::chrono::duration<double, std::milli>(finished_at - request_started).count();
    const double decode_seconds = std::chrono::duration<double>(finished_at - first_token_at).count();
    const double decode_tokens_per_second = output.size() > 1 && decode_seconds > 0 ? static_cast<double>(output.size() - 1) / decode_seconds : 0.0;
    write_result(options, static_cast<uint32_t>(prompt.size()), output, proposals, accepted, rejected, worker_decode_calls, wire_frames, ttft_ms, total_ms, decode_tokens_per_second);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::strcmp(argv[1], "--help") == 0)) {
            print_usage(argv[0]);
            return argc == 1 ? 1 : 0;
        }
        const Options options = parse_args(argc, argv);
        llama_backend_init();
        ggml_backend_load_all();
        Model model = load_model(options.model_path, options.gpu_layers, options.mode == "worker" ? options.split : 0, options.mode == "single" ? 0 : options.mode == "worker" ? UINT32_MAX : options.split);
        const uint32_t layers = static_cast<uint32_t>(llama_model_n_layer(model.get()));
        if (options.mode == "single") {
            run_single(options, model.get());
        } else {
            if (options.split >= layers) { throw std::runtime_error("--split must be less than model layer count"); }
            const auto local = handshake_for(options, model.get());
            if (options.mode == "worker") { run_worker(options, model.get(), local); }
            else { run_coordinator(options, model.get(), local); }
        }
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "split-lookahead: %s\n", error.what());
        return 1;
    }
}
