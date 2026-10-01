#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static int gpu_layers = 0;

static model_ptr load(const char * path, int start, int end) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    params.stage_layer_start = start;
    params.stage_layer_end = end;
    return model_ptr(llama_model_load_from_file(path, params), llama_model_free);
}

static context_ptr context(llama_model * model, bool prefix) {
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 2;
    params.n_ubatch = 2;
    params.n_seq_max = 2;
    params.n_threads = 2;
    params.n_threads_batch = 2;
    params.kv_unified = true;
    params.embeddings = prefix;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_ptr result(llama_init_from_model(model, params), llama_free);
    require(bool(result), "context creation failed");
    return result;
}

static std::vector<float> step(llama_context * ctx, llama_token token, int position,
                               bool prefix, int width, llama_seq_id seq = 0,
                               std::vector<float> * activation = nullptr) {
    llama_pos pos[4] = {position, position, position, 0};
    int32_t count = 1;
    llama_seq_id * seqs = &seq;
    int8_t output = 1;
    llama_batch batch = {};
    batch.n_tokens = 1;
    batch.token = activation ? nullptr : &token;
    batch.embd = activation ? activation->data() : nullptr;
    batch.pos = pos;
    batch.n_seq_id = &count;
    batch.seq_id = &seqs;
    batch.logits = &output;
    require(llama_decode(ctx, batch) == 0, "decode failed");
    const float * data = prefix ? llama_get_embeddings_ith(ctx, 0) : llama_get_logits_ith(ctx, 0);
    require(data != nullptr, "missing output");
    return std::vector<float>(data, data + width);
}

static void equal(const std::vector<float> & actual, const std::vector<float> & expected, const char * message) {
    require(actual.size() == expected.size(), "output shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || actual[i] != expected[i]) {
            fprintf(stderr, "%s at %zu: %.9g != %.9g\n", message, i, actual[i], expected[i]);
            throw std::runtime_error(message);
        }
    }
}

static void test_partition(const char * path, int split) {
    auto full_model = load(path, 0, 0);
    require(bool(full_model), "full model load failed");
    int layers = llama_model_n_layer(full_model.get());
    int hidden = llama_model_n_embd(full_model.get());
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(full_model.get()));
    auto prefix_model = load(path, 0, split);
    auto suffix_model = load(path, split, layers);
    require(bool(prefix_model) && bool(suffix_model), "stage model load failed");
    auto full = context(full_model.get(), false);
    auto prefix = context(prefix_model.get(), true);
    auto reference = context(prefix_model.get(), true);
    auto suffix = context(suffix_model.get(), false);

    const std::vector<llama_token> tokens = {3, 17, 9, 31, 6, 15, 42, 27, 5, 12, 19, 23};
    for (int p = 0; p < 4; ++p) {
        auto activation = step(prefix.get(), tokens[p], p, true, hidden);
        require(llama_get_logits_ith(prefix.get(), 0) == nullptr, "prefix exposed logits");
        equal(activation, step(reference.get(), tokens[p], p, true, hidden), "prefix prefill");
        equal(step(suffix.get(), 0, p, false, vocab, 0, &activation),
              step(full.get(), tokens[p], p, false, vocab), "full vs staged logits");
    }
    auto memory = llama_get_memory(prefix.get());
    for (int p = 4; p < int(tokens.size()); ++p) {
        llama_memory_seq_cp(memory, 0, 1, -1, -1);
        const bool accept = p % 2 == 0;
        auto branch = step(prefix.get(), accept ? tokens[p] : (tokens[p] + 1) % vocab, p, true, hidden, 1);
        require(llama_memory_seq_pos_max(memory, 0) == p - 1, "branch modified base position");
        std::vector<float> activation;
        if (accept) {
            require(llama_memory_seq_rm(memory, 0, -1, -1), "base clear failed");
            llama_memory_seq_cp(memory, 1, 0, -1, -1);
            require(llama_memory_seq_rm(memory, 1, -1, -1), "accepted branch clear failed");
            activation = std::move(branch);
        } else {
            require(llama_memory_seq_rm(memory, 1, -1, -1), "discard failed");
            activation = step(prefix.get(), tokens[p], p, true, hidden);
        }
        equal(activation, step(reference.get(), tokens[p], p, true, hidden), "transaction output");
        equal(step(suffix.get(), 0, p, false, vocab, 0, &activation),
              step(full.get(), tokens[p], p, false, vocab), "transaction logits");
        require(llama_memory_seq_pos_max(memory, 0) == p, "wrong committed position");
        require(llama_memory_seq_pos_max(memory, 1) == -1, "branch survived settlement");
    }
    llama_memory_clear(memory, true);
    llama_memory_clear(llama_get_memory(reference.get()), true);
    for (int p = 0; p < 4; ++p) {
        equal(step(prefix.get(), tokens[p], p, true, hidden),
              step(reference.get(), tokens[p], p, true, hidden), "reset replay");
    }
    fprintf(stdout, "partition %d/%d: exact logits, accept/discard, positions and reset passed\n", split, layers - split);
}

int main(int argc, char ** argv) {
    if (argc != 2 && (argc != 4 || std::string(argv[2]) != "--gpu-layers")) {
        fprintf(stderr, "usage: %s four-layer-qwen35.gguf [--gpu-layers N]\n", argv[0]);
        return 2;
    }
    if (argc == 4) {
        errno = 0;
        char * end = nullptr;
        const long value = std::strtol(argv[3], &end, 10);
        if (errno != 0 || end == argv[3] || *end != '\0' || value < 0 || value > INT_MAX) {
            fprintf(stderr, "invalid --gpu-layers value: %s\n", argv[3]);
            return 2;
        }
        gpu_layers = static_cast<int>(value);
    }
    ggml_backend_load_all();
    llama_backend_init();
    try {
        auto model = load(argv[1], 0, 0);
        require(bool(model), "fixture failed to load");
        require(llama_model_n_layer(model.get()) == 4, "expected four-layer fixture");
        require(!load(argv[1], 3, 2), "reversed range accepted");
        require(!load(argv[1], 0, 5), "out of range stage accepted");
        for (int split : {2, 1, 3}) {
            test_partition(argv[1], split);
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
