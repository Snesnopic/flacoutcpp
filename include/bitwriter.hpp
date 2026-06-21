#ifndef BITWRITER_HPP
#define BITWRITER_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <array>

// Writes bits MSB-first into a byte buffer.
// Used by FrameWriter to construct FLAC frames bit by bit.
class BitWriter {
public:
    BitWriter() = default;

    // Write n bits of value, MSB-first. n must be 1..64.
    void write_bits(uint64_t value, int n);

    // Write a signed value in two's complement, MSB-first. n must be 1..32.
    void write_signed_bits(int32_t value, int n);

    // Write value in unary: (value) zeros followed by a 1.
    void write_unary(uint32_t value);

    // Rice-encode one signed residual sample with parameter k.
    void write_rice_sample(int32_t sample, int k);

    // Write a 64-bit integer in the FLAC UTF-8-like variable-length encoding
    // (used for frame/sample numbers in FLAC frame headers).
    void write_utf8(uint64_t value);

    // Pad to the next byte boundary with zero bits.
    void align();

    // Return the number of complete + in-progress bytes.
    size_t byte_size() const;

    // Return the total number of bits written so far.
    size_t bit_count() const { return m_buf.size() * 8 + (size_t)m_bits; }

    // Read-only view of the accumulated byte buffer.
    const std::vector<uint8_t>& buffer() const { return m_buf; }

    // Compute CRC-8  (FLAC poly: x^8+x^2+x+1 = 0x07) over byte range [from, to).
    uint8_t  crc8 (size_t from_byte, size_t to_byte) const;

    // Compute CRC-16 (FLAC poly: x^16+x^15+x^2+1 = 0x8005) over byte range [from, to).
    uint16_t crc16(size_t from_byte, size_t to_byte) const;

    // Reset to empty state.
    void reset();

private:
    std::vector<uint8_t> m_buf;
    uint8_t m_partial = 0;  // partially-filled current byte
    int     m_bits    = 0;  // valid bits in m_partial (0–7)

    // Lookup tables built at compile time (C++17 constexpr).
    static const std::array<uint8_t,  256> CRC8_TABLE;
    static const std::array<uint16_t, 256> CRC16_TABLE;
};

#endif // BITWRITER_HPP
