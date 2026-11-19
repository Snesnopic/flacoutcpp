#ifndef MESSAGES_HPP
#define MESSAGES_HPP

// Routes the library's messages to a caller's MessageSink, or prints them the
// way the CLI expects when there is none.

#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "optimizer.hpp"

namespace flacoutcpp::detail {

// The sink is only called on the thread that created this object; a message
// raised on any other thread waits in a queue until that thread calls flush().
// Without a sink, errors and warnings go to stderr and info to stdout, at once.
class Messages {
public:
    explicit Messages(MessageSink sink = {});
    Messages(const Messages&) = delete;
    Messages& operator=(const Messages&) = delete;

    void report(MessageLevel level, std::string text);
    void flush();

    // the running "\r  done/total" counter: a terminal display, never sent to a sink
    void progress(std::size_t done, std::size_t total, bool percent);
    void end_progress();

private:
    MessageSink     m_sink;
    std::thread::id m_owner;
    std::mutex      m_mutex;
    std::vector<std::pair<MessageLevel, std::string>> m_pending;
};

} // namespace flacoutcpp::detail

#endif // MESSAGES_HPP
