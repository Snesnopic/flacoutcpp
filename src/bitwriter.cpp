#include "bitwriter.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

// ---------------------------------------------------------------------------
// Compile-time CRC table generation (C++17 constexpr)
// ---------------------------------------------------------------------------

static constexpr std::array<uint8_t, 256> make_crc8_table() {
    std::array<uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        uint8_t crc = (uint8_t)i;
        for (int j = 0; j < 8; ++j)
            crc = (crc & 0x80u) ? (uint8_t)((crc << 1) ^ 0x07u) : (uint8_t)(crc << 1);
        t[i] = crc;
    }
    return t;
}

static constexpr std::array<uint16_t, 256> make_crc16_table() {
    std::array<uint16_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        uint16_t crc = (uint16_t)(i << 8);
        for (int j = 0; j < 8; ++j)
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x8005u) : (uint16_t)(crc << 1);
        t[i] = crc;
    }
    return t;
}

const std::array<uint8_t,  256> BitWriter::CRC8_TABLE  = make_crc8_table();
const std::array<uint16_t, 256> BitWriter::CRC16_TABLE = make_crc16_table();

// ---------------------------------------------------------------------------
// Core write primitives
// ---------------------------------------------------------------------------

void BitWriter::write_bits(uint64_t value, int n) {
    assert(n >= 1 && n <= 64);
    if (n < 64) value &= (1ULL << n) - 1ULL;

    while (n > 0) {
        int free = 8 - m_bits;
        int take = (n < free) ? n : free;
        // Extract the top `take` bits of value
        int shift = n - take;
        uint8_t chunk = (uint8_t)((value >> shift) & ((1u << take) - 1u));
        m_partial = (uint8_t)((m_partial << take) | chunk);
        m_bits += take;
        n     -= take;
        if (m_bits == 8) {
            m_buf.push_back(m_partial);
            m_partial = 0;
            m_bits    = 0;
        }
    }
}

void BitWriter::write_signed_bits(int32_t value, int n) {
    write_bits((uint64_t)(uint32_t)value, n);
}

void BitWriter::write_unary(uint32_t value) {
    // Write `value` zero bits, then a terminating 1.
    while (value >= 64) { write_bits(0ULL, 64); value -= 64; }
    if (value > 0)       write_bits(0ULL, (int)value);
    write_bits(1ULL, 1);
}

void BitWriter::write_rice_sample(int32_t sample, int k) {
    // Zigzag (fold) signed → unsigned
    uint32_t u = (uint32_t)((sample << 1) ^ (sample >> 31));
    uint32_t q = u >> k;          // quotient
    uint32_t r = u & ((1u << k) - 1u); // remainder (only valid when k>0)
    write_unary(q);
    if (k > 0) write_bits(r, k);
}

void BitWriter::write_utf8(uint64_t v) {
    // FLAC uses a UTF-8-like variable-length encoding for sample numbers (up to 36 bits).
    if (v < 0x80ULL) {
        write_bits(v, 8);
    } else if (v < 0x800ULL) {
        write_bits(0xC0ULL | (v >> 6),  8);
        write_bits(0x80ULL | (v & 0x3F), 8);
    } else if (v < 0x10000ULL) {
        write_bits(0xE0ULL | (v >> 12),         8);
        write_bits(0x80ULL | ((v >>  6) & 0x3F), 8);
        write_bits(0x80ULL | ( v        & 0x3F), 8);
    } else if (v < 0x200000ULL) {
        write_bits(0xF0ULL | (v >> 18),          8);
        write_bits(0x80ULL | ((v >> 12) & 0x3F), 8);
        write_bits(0x80ULL | ((v >>  6) & 0x3F), 8);
        write_bits(0x80ULL | ( v        & 0x3F), 8);
    } else if (v < 0x4000000ULL) {
        write_bits(0xF8ULL | (v >> 24),          8);
        write_bits(0x80ULL | ((v >> 18) & 0x3F), 8);
        write_bits(0x80ULL | ((v >> 12) & 0x3F), 8);
        write_bits(0x80ULL | ((v >>  6) & 0x3F), 8);
        write_bits(0x80ULL | ( v        & 0x3F), 8);
    } else if (v < 0x80000000ULL) {
        write_bits(0xFCULL | (v >> 30),          8);
        write_bits(0x80ULL | ((v >> 24) & 0x3F), 8);
        write_bits(0x80ULL | ((v >> 18) & 0x3F), 8);
        write_bits(0x80ULL | ((v >> 12) & 0x3F), 8);
        write_bits(0x80ULL | ((v >>  6) & 0x3F), 8);
        write_bits(0x80ULL | ( v        & 0x3F), 8);
    } else {
        // 7-byte form: handles values up to 2^36-1
        write_bits(0xFEULL,                        8);
        write_bits(0x80ULL | ((v >> 30) & 0x3F),  8);
        write_bits(0x80ULL | ((v >> 24) & 0x3F),  8);
        write_bits(0x80ULL | ((v >> 18) & 0x3F),  8);
        write_bits(0x80ULL | ((v >> 12) & 0x3F),  8);
        write_bits(0x80ULL | ((v >>  6) & 0x3F),  8);
        write_bits(0x80ULL | ( v        & 0x3F),  8);
    }
}

void BitWriter::align() {
    if (m_bits > 0) {
        // Pad with zeros to complete the byte
        m_partial = (uint8_t)(m_partial << (8 - m_bits));
        m_buf.push_back(m_partial);
        m_partial = 0;
        m_bits    = 0;
    }
}

size_t BitWriter::byte_size() const {
    return m_buf.size() + (m_bits > 0 ? 1 : 0);
}

void BitWriter::reset() {
    m_buf.clear();
    m_partial = 0;
    m_bits    = 0;
}

// ---------------------------------------------------------------------------
// CRC computation
// ---------------------------------------------------------------------------

uint8_t BitWriter::crc8(size_t from, size_t to) const {
    uint8_t crc = 0;
    for (size_t i = from; i < to && i < m_buf.size(); ++i)
        crc = CRC8_TABLE[(crc ^ m_buf[i]) & 0xFF];
    return crc;
}

uint16_t BitWriter::crc16(size_t from, size_t to) const {
    uint16_t crc = 0;
    for (size_t i = from; i < to && i < m_buf.size(); ++i)
        crc = (uint16_t)((crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ m_buf[i]) & 0xFF]);
    return crc;
}
