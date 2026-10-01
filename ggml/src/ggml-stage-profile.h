#pragma once

// Opt-in, process-local trace for diagnosing scheduler/RPC/CUDA stage boundaries.
// GGML_STAGE_PROFILE_JSONL must name an owned output file. This header is kept
// private to ggml so normal builds and public ABI remain unchanged.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ggml_stage_profile {

inline bool enabled() {
    static const bool active = [] {
        const char * path = std::getenv("GGML_STAGE_PROFILE_JSONL");
        return path != nullptr && path[0] == '/';
    }();
    return active;
}

inline uint64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline uint64_t unix_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string quote(const char * value) {
    std::string out = "\"";
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(*p));
        } else if (*p < 0x20) {
            const char hex[] = "0123456789abcdef";
            out += "\\u00";
            out.push_back(hex[*p >> 4]);
            out.push_back(hex[*p & 15]);
        } else {
            out.push_back(static_cast<char>(*p));
        }
    }
    out.push_back('"');
    return out;
}

inline void emit(const char * event, const std::string & fields) {
    if (!enabled()) {
        return;
    }
#if !defined(_WIN32)
    static const int fd = [] {
        const char * path = std::getenv("GGML_STAGE_PROFILE_JSONL");
        return open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    }();
    if (fd < 0) {
        return;
    }
    static std::mutex write_mutex;
    std::ostringstream line;
    line << "{\"event\":" << quote(event)
         << ",\"monotonic_ns\":" << monotonic_ns()
         << ",\"unix_ns\":" << unix_ns()
         << ",\"pid\":" << getpid();
    if (!fields.empty()) {
        line << ',' << fields;
    }
    line << "}\n";
    const std::string payload = line.str();
    std::lock_guard<std::mutex> lock(write_mutex);
    // One append write per record prevents interleaving between translation units.
    const ssize_t written = write(fd, payload.data(), payload.size());
    if (written < 0 || static_cast<size_t>(written) != payload.size()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "GGML stage profile JSONL write was incomplete\n");
        }
    }
#else
    (void) event;
    (void) fields;
#endif
}

} // namespace ggml_stage_profile
