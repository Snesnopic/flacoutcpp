#include "messages.hpp"
#include <iostream>

namespace flacoutcpp::detail {

Messages::Messages(MessageSink sink)
    : m_sink(std::move(sink)), m_owner(std::this_thread::get_id()) {}

void Messages::report(MessageLevel level, std::string text)
{
    if (!m_sink) {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (level == MessageLevel::info)
            std::cout << text << "\n";
        else
            std::cerr << (level == MessageLevel::error ? "Error: " : "Warning: ") << text << "\n";
        return;
    }
    if (std::this_thread::get_id() != m_owner) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_pending.emplace_back(level, std::move(text));
        return;
    }
    // what workers queued happened first
    flush();
    m_sink(level, text);
}

void Messages::flush()
{
    std::vector<std::pair<MessageLevel, std::string>> batch;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        batch.swap(m_pending);
    }
    for (const auto& [level, text] : batch) m_sink(level, text);
}

void Messages::progress(std::size_t done, std::size_t total, bool percent)
{
    if (m_sink) return;
    std::lock_guard<std::mutex> lk(m_mutex);
    std::cout << "\r  " << done << "/" << total;
    if (percent) std::cout << " (" << done * 100 / total << "%)";
    std::cout << std::flush;
}

void Messages::end_progress()
{
    if (!m_sink) std::cout << "\n";
}

} // namespace flacoutcpp::detail
