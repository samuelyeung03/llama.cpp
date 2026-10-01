#pragma once
// Bounded, single-use snapshots. No GPU pointer is exposed to a peer connection.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#include <cerrno>
#include <sys/random.h>

namespace hoshino_rpc {
using ticket = std::array<uint8_t, 32>;
class peer_exports {
public:
    using clock = std::chrono::steady_clock;
    static constexpr size_t max_tensor = 32 * 1024 * 1024;
    static constexpr size_t max_bytes = 64 * 1024 * 1024;
    static constexpr size_t max_entries = 16;
    static constexpr auto ttl = std::chrono::seconds(300);

    bool publish(uint64_t owner, std::vector<uint8_t> data, ticket & key, clock::time_point now = clock::now()) {
        if (!owner || data.empty() || data.size() > max_tensor) return false;
        size_t filled = 0;
        while (filled < key.size()) {
            ssize_t n = getrandom(key.data() + filled, key.size() - filled, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            filled += static_cast<size_t>(n);
        }
        std::lock_guard<std::mutex> lock(mutex);
        prune(now);
        if (entries.size() >= max_entries || data.size() > max_bytes - bytes || entries.count(key)) return false;
        bytes += data.size();
        entries.emplace(key, entry{owner, now + ttl, std::move(data)});
        return true;
    }
    bool take(const ticket & key, size_t expected_size, std::vector<uint8_t> & data, clock::time_point now = clock::now()) {
        std::lock_guard<std::mutex> lock(mutex);
        prune(now);
        auto it = entries.find(key);
        if (it == entries.end() || expected_size != it->second.data.size()) return false;
        bytes -= it->second.data.size();
        data = std::move(it->second.data);
        entries.erase(it); // Replay/partial-transfer retry is terminal, never a second copy.
        return true;
    }
    void clear_owner(uint64_t owner) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->second.owner == owner) { bytes -= it->second.data.size(); it = entries.erase(it); }
            else ++it;
        }
    }
private:
    struct entry { uint64_t owner; clock::time_point expires; std::vector<uint8_t> data; };
    void prune(clock::time_point now) {
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->second.expires <= now) { bytes -= it->second.data.size(); it = entries.erase(it); }
            else ++it;
        }
    }
    std::mutex mutex;
    std::map<ticket, entry> entries;
    size_t bytes = 0;
};
} // namespace hoshino_rpc
