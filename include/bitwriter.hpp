/**
 * @file bitwriter.hpp
 * @brief High-performance MSB-first bitwise stream writer.
 */

#ifndef BITWRITER_HPP
#define BITWRITER_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <array>

/**
 * @brief Writes bits MSB-first into a dynamically growing byte buffer.
 * 
 * Used internally by FrameWriter to construct FLAC frame headers and
 * payload sections bit-by-bit. Also provides fast CRC-8 and CRC-16
 * checksum generation required by the FLAC specification.
 */
class BitWriter {
public:
    /**
     * @brief Construct an empty BitWriter.
     */
    BitWriter() = default;

    /**
     * @brief Write n bits of an unsigned integer, MSB-first.
     * 
     * @param value The value containing the bits to write.
     * @param n     The number of bits to write (1..64).
     */
    void write_bits(uint64_t value, int n);

    /**
     * @brief Write a signed value in two's complement, MSB-first.
     * 
     * @param value The signed value to write.
     * @param n     The number of bits to write (1..32).
     */
    void write_signed_bits(int32_t value, int n);

    /**
     * @brief Write a value in unary code: @c value zeroes followed by a 1.
     * 
     * @param value The number of zeroes to write before the terminating 1.
     */
    void write_unary(uint32_t value);

    /**
     * @brief Rice-encode one signed residual sample with partition parameter k.
     * 
     * @param sample The signed residual sample.
     * @param k      The Rice parameter (number of LSBs sent verbatim).
     */
    void write_rice_sample(int32_t sample, int k);

    /**
     * @brief Write a 64-bit integer using FLAC's UTF-8-like variable-length encoding.
     * 
     * This is used for absolute frame and sample numbers in FLAC headers.
     * 
     * @param value The value to encode.
     */
    void write_utf8(uint64_t value);

    /**
     * @brief Pad the stream to the next byte boundary with zero bits.
     * 
     * If the stream is already byte-aligned, this does nothing.
     */
    void align();

    /**
     * @brief Return the number of complete and in-progress bytes.
     * 
     * @return size_t Byte count.
     */
    size_t byte_size() const;

    /**
     * @brief Return the total number of bits written so far.
     * 
     * @return size_t Bit count.
     */
    size_t bit_count() const { return m_buf.size() * 8 + (size_t)m_bits; }

    /**
     * @brief Get a read-only view of the accumulated byte buffer.
     * 
     * @return const std::vector<uint8_t>& The underlying byte storage.
     */
    const std::vector<uint8_t>& buffer() const { return m_buf; }

    /**
     * @brief Compute CRC-8 over a specified byte range.
     * 
     * FLAC polynomial: x^8 + x^2 + x + 1 (0x07).
     * 
     * @param from_byte Starting byte index.
     * @param to_byte   Ending byte index (exclusive).
     * @return uint8_t  The computed CRC-8 checksum.
     */
    uint8_t  crc8 (size_t from_byte, size_t to_byte) const;

    /**
     * @brief Compute CRC-16 over a specified byte range.
     * 
     * FLAC polynomial: x^16 + x^15 + x^2 + 1 (0x8005).
     * 
     * @param from_byte Starting byte index.
     * @param to_byte   Ending byte index (exclusive).
     * @return uint16_t The computed CRC-16 checksum.
     */
    uint16_t crc16(size_t from_byte, size_t to_byte) const;

    /**
     * @brief Reset the writer to its empty initial state.
     */
    void reset();

private:
    /// @cond INTERNAL

    std::vector<uint8_t> m_buf;
    uint8_t m_partial = 0;  // partially-filled current byte
    int     m_bits    = 0;  // valid bits in m_partial (0–7)

    // Lookup tables built at compile time (C++17 constexpr).
    static const std::array<uint8_t,  256> CRC8_TABLE;
    static const std::array<uint16_t, 256> CRC16_TABLE;

    /// @endcond
};

#endif // BITWRITER_HPP
