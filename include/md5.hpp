#ifndef MD5_HPP
#define MD5_HPP

// Minimal portable RFC 1321 MD5 implementation.
// Used to compute the FLAC STREAMINFO audio signature.
// No external dependencies.

#include <array>
#include <cstdint>
#include <cstring>

// NOLINTBEGIN

class MD5 {
public:
    MD5() { reset(); }

    void reset() {
        m_state[0] = 0x67452301u;
        m_state[1] = 0xEFCDAB89u;
        m_state[2] = 0x98BADCFEu;
        m_state[3] = 0x10325476u;
        m_count[0] = m_count[1] = 0;
        std::memset(m_buf, 0, sizeof(m_buf));
    }

    // Feed raw bytes into the MD5 context.
    void update(const uint8_t* data, size_t len) {
        // Number of bytes already in the partial block
        uint32_t idx = (m_count[0] >> 3) & 0x3Fu;

        // Update bit count
        m_count[0] += (uint32_t)(len << 3);
        if (m_count[0] < (uint32_t)(len << 3)) ++m_count[1];
        m_count[1] += (uint32_t)(len >> 29);

        uint32_t part = 64u - idx;
        size_t   i    = 0;

        if (len >= part) {
            std::memcpy(&m_buf[idx], data, part);
            transform(m_buf);
            for (i = part; i + 63 < len; i += 64)
                transform(&data[i]);
            idx = 0;
        }

        std::memcpy(&m_buf[idx], &data[i], len - i);
    }

    // Finalise and return the 16-byte digest.
    std::array<uint8_t, 16> digest() {
        static const uint8_t PADDING[64] = { 0x80 };
        uint8_t bits[8];
        encode(bits, m_count, 8);

        // Pad to 56 bytes mod 64
        uint32_t idx  = (m_count[0] >> 3) & 0x3Fu;
        uint32_t plen = (idx < 56) ? (56 - idx) : (120 - idx);
        update(PADDING, plen);
        update(bits, 8);

        std::array<uint8_t, 16> out;
        encode(out.data(), m_state, 16);
        return out;
    }

private:
    uint32_t m_state[4];
    uint32_t m_count[2];
    uint8_t  m_buf[64];

    static uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
    static uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
    static uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
    static uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
    static uint32_t ROL(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

    static void encode(uint8_t* out, const uint32_t* in, size_t len) {
        for (size_t i = 0, j = 0; j < len; ++i, j += 4) {
            out[j]   = (uint8_t)(in[i]);
            out[j+1] = (uint8_t)(in[i] >>  8);
            out[j+2] = (uint8_t)(in[i] >> 16);
            out[j+3] = (uint8_t)(in[i] >> 24);
        }
    }

    static void decode(uint32_t* out, const uint8_t* in, size_t len) {
        for (size_t i = 0, j = 0; j < len; ++i, j += 4)
            out[i] = (uint32_t)in[j] | ((uint32_t)in[j+1] << 8)
                   | ((uint32_t)in[j+2] << 16) | ((uint32_t)in[j+3] << 24);
    }

    void transform(const uint8_t block[64]) {
        uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3], x[16];
        decode(x, block, 64);

        // Round 1
        auto FF = [&](uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                      uint32_t xk, uint32_t s, uint32_t t) {
            a = b + ROL(a + F(b,c,d) + xk + t, s); };
        FF(a,b,c,d, x[ 0],  7, 0xD76AA478U); FF(d,a,b,c, x[ 1], 12, 0xE8C7B756U);
        FF(c,d,a,b, x[ 2], 17, 0x242070DBU); FF(b,c,d,a, x[ 3], 22, 0xC1BDCEEEU);
        FF(a,b,c,d, x[ 4],  7, 0xF57C0FAFU); FF(d,a,b,c, x[ 5], 12, 0x4787C62AU);
        FF(c,d,a,b, x[ 6], 17, 0xA8304613U); FF(b,c,d,a, x[ 7], 22, 0xFD469501U);
        FF(a,b,c,d, x[ 8],  7, 0x698098D8U); FF(d,a,b,c, x[ 9], 12, 0x8B44F7AFU);
        FF(c,d,a,b, x[10], 17, 0xFFFF5BB1U); FF(b,c,d,a, x[11], 22, 0x895CD7BEU);
        FF(a,b,c,d, x[12],  7, 0x6B901122U); FF(d,a,b,c, x[13], 12, 0xFD987193U);
        FF(c,d,a,b, x[14], 17, 0xA679438EU); FF(b,c,d,a, x[15], 22, 0x49B40821U);

        // Round 2
        auto GG = [&](uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                      uint32_t xk, uint32_t s, uint32_t t) {
            a = b + ROL(a + G(b,c,d) + xk + t, s); };
        GG(a,b,c,d, x[ 1],  5, 0xF61E2562U); GG(d,a,b,c, x[ 6],  9, 0xC040B340U);
        GG(c,d,a,b, x[11], 14, 0x265E5A51U); GG(b,c,d,a, x[ 0], 20, 0xE9B6C7AAU);
        GG(a,b,c,d, x[ 5],  5, 0xD62F105DU); GG(d,a,b,c, x[10],  9, 0x02441453U);
        GG(c,d,a,b, x[15], 14, 0xD8A1E681U); GG(b,c,d,a, x[ 4], 20, 0xE7D3FBC8U);
        GG(a,b,c,d, x[ 9],  5, 0x21E1CDE6U); GG(d,a,b,c, x[14],  9, 0xC33707D6U);
        GG(c,d,a,b, x[ 3], 14, 0xF4D50D87U); GG(b,c,d,a, x[ 8], 20, 0x455A14EDU);
        GG(a,b,c,d, x[13],  5, 0xA9E3E905U); GG(d,a,b,c, x[ 2],  9, 0xFCEFA3F8U);
        GG(c,d,a,b, x[ 7], 14, 0x676F02D9U); GG(b,c,d,a, x[12], 20, 0x8D2A4C8AU);

        // Round 3
        auto HH = [&](uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                      uint32_t xk, uint32_t s, uint32_t t) {
            a = b + ROL(a + H(b,c,d) + xk + t, s); };
        HH(a,b,c,d, x[ 5],  4, 0xFFFA3942U); HH(d,a,b,c, x[ 8], 11, 0x8771F681U);
        HH(c,d,a,b, x[11], 16, 0x6D9D6122U); HH(b,c,d,a, x[14], 23, 0xFDE5380CU);
        HH(a,b,c,d, x[ 1],  4, 0xA4BEEA44U); HH(d,a,b,c, x[ 4], 11, 0x4BDECFA9U);
        HH(c,d,a,b, x[ 7], 16, 0xF6BB4B60U); HH(b,c,d,a, x[10], 23, 0xBEBFBC70U);
        HH(a,b,c,d, x[13],  4, 0x289B7EC6U); HH(d,a,b,c, x[ 0], 11, 0xEAA127FAU);
        HH(c,d,a,b, x[ 3], 16, 0xD4EF3085U); HH(b,c,d,a, x[ 6], 23, 0x04881D05U);
        HH(a,b,c,d, x[ 9],  4, 0xD9D4D039U); HH(d,a,b,c, x[12], 11, 0xE6DB99E5U);
        HH(c,d,a,b, x[15], 16, 0x1FA27CF8U); HH(b,c,d,a, x[ 2], 23, 0xC4AC5665U);

        // Round 4
        auto II = [&](uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                      uint32_t xk, uint32_t s, uint32_t t) {
            a = b + ROL(a + I(b,c,d) + xk + t, s); };
        II(a,b,c,d, x[ 0],  6, 0xF4292244U); II(d,a,b,c, x[ 7], 10, 0x432AFF97U);
        II(c,d,a,b, x[14], 15, 0xAB9423A7U); II(b,c,d,a, x[ 5], 21, 0xFC93A039U);
        II(a,b,c,d, x[12],  6, 0x655B59C3U); II(d,a,b,c, x[ 3], 10, 0x8F0CCC92U);
        II(c,d,a,b, x[10], 15, 0xFFEFF47DU); II(b,c,d,a, x[ 1], 21, 0x85845DD1U);
        II(a,b,c,d, x[ 8],  6, 0x6FA87E4FU); II(d,a,b,c, x[15], 10, 0xFE2CE6E0U);
        II(c,d,a,b, x[ 6], 15, 0xA3014314U); II(b,c,d,a, x[13], 21, 0x4E0811A1U);
        II(a,b,c,d, x[ 4],  6, 0xF7537E82U); II(d,a,b,c, x[11], 10, 0xBD3AF235U);
        II(c,d,a,b, x[ 2], 15, 0x2AD7D2BBU); II(b,c,d,a, x[ 9], 21, 0xEB86D391U);

        m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
    }
};

// NOLINTEND

#endif // MD5_HPP
