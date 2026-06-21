#include "frame_writer.hpp"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdexcept>

// ============================================================
// Helpers: encode frame-header variable-length fields
// ============================================================

// FLAC block-size codes (4 bits in frame header).
// Returns the 4-bit code; also writes the optional trailing byte(s) into bw_extra.
static uint8_t blocksize_code(uint32_t bs, uint32_t& extra_val, int& extra_bits) {
    extra_bits = 0;
    switch (bs) {
        case 192:   return 0x1;
        case 576:   return 0x2;
        case 1152:  return 0x3;
        case 2304:  return 0x4;
        case 4608:  return 0x5;
        case 256:   return 0x8;
        case 512:   return 0x9;
        case 1024:  return 0xA;
        case 2048:  return 0xB;
        case 4096:  return 0xC;
        case 8192:  return 0xD;
        case 16384: return 0xE;
        case 32768: return 0xF;
        default:
            if (bs <= 256) {
                extra_val  = bs - 1;   // 8-bit
                extra_bits = 8;
                return 0x6;
            } else {
                extra_val  = bs - 1;   // 16-bit
                extra_bits = 16;
                return 0x7;
            }
    }
}

static uint8_t samplerate_code(uint32_t sr, uint32_t& extra_val, int& extra_bits) {
    extra_bits = 0;
    switch (sr) {
        case 88200:  return 0x1;
        case 176400: return 0x2;
        case 192000: return 0x3;
        case 8000:   return 0x4;
        case 16000:  return 0x5;
        case 22050:  return 0x6;
        case 24000:  return 0x7;
        case 32000:  return 0x8;
        case 44100:  return 0x9;
        case 48000:  return 0xA;
        case 96000:  return 0xB;
        default:
            if (sr % 1000 == 0 && sr / 1000 <= 255) {
                extra_val = sr / 1000; extra_bits = 8; return 0xC;
            } else if (sr <= 65535) {
                extra_val = sr; extra_bits = 16; return 0xD;
            } else if (sr % 10 == 0 && sr / 10 <= 65535) {
                extra_val = sr / 10; extra_bits = 16; return 0xE;
            }
            return 0x0; // get from STREAMINFO (shouldn't happen)
    }
}

uint8_t FrameWriter::encode_bps(uint32_t bps) {
    switch (bps) {
        case 8:  return 0x1;
        case 12: return 0x2;
        case 16: return 0x4;
        case 20: return 0x5;
        case 24: return 0x6;
        case 32: return 0x7;
        default: return 0x0; // get from STREAMINFO
    }
}

// ============================================================
// Main frame serialization
// ============================================================

std::vector<uint8_t> FrameWriter::write_frame(
    const BlockParams&                        params,
    const std::vector<std::vector<int32_t>>& pcm_data,
    uint64_t                                  sample_number,
    uint32_t                                  sample_rate,
    uint32_t                                  bps)
{
    const uint32_t nch      = (uint32_t)pcm_data.size();
    const uint32_t bsize    = params.block_size;
    const int      sm       = params.stereo_mode;

    // --- Build transformed channel buffers ---------------------------------
    // For joint-stereo modes we must encode the *decorrelated* signal,
    // but the PCM stored in pcm_data is always L/R.
    std::vector<std::vector<int32_t>> ch(nch);
    uint32_t bps_ch[2] = { bps, bps };   // effective bps per channel

    if (nch == 2 && sm != 0) {
        const auto& L = pcm_data[0];
        const auto& R = pcm_data[1];
        ch[0].resize(bsize);
        ch[1].resize(bsize);
        for (uint32_t i = 0; i < bsize; ++i) {
            int32_t l = L[sample_number + i];
            int32_t r = R[sample_number + i];
            if (sm == 8) {          // Left-Side
                ch[0][i] = l;
                ch[1][i] = l - r;
                bps_ch[1] = bps + 1;
            } else if (sm == 9) {   // Right-Side
                ch[0][i] = l - r;
                ch[1][i] = r;
                bps_ch[0] = bps + 1;
            } else {                // Mid-Side (10)
                ch[0][i] = (l + r) >> 1;
                ch[1][i] = l - r;
                bps_ch[1] = bps + 1;
            }
        }
    } else {
        for (uint32_t c = 0; c < nch; ++c) {
            ch[c].assign(pcm_data[c].begin() + sample_number,
                         pcm_data[c].begin() + sample_number + bsize);
        }
    }

    // --- Build frame -------------------------------------------------------
    BitWriter bw;
    const size_t frame_start = 0;

    // Sync + reserved(0) + blocking_strategy(1=variable)
    bw.write_bits(0x3FFE, 14);
    bw.write_bits(0, 1);
    bw.write_bits(1, 1);  // variable blocking strategy → sample-number addressing

    // Block size (4 bits header code + optional trailing bytes)
    uint32_t bs_extra_val = 0; int bs_extra_bits = 0;
    uint8_t  bs_code = blocksize_code(bsize, bs_extra_val, bs_extra_bits);
    bw.write_bits(bs_code, 4);

    // Sample rate (4 bits + optional trailing bytes)
    uint32_t sr_extra_val = 0; int sr_extra_bits = 0;
    uint8_t  sr_code = samplerate_code(sample_rate, sr_extra_val, sr_extra_bits);
    bw.write_bits(sr_code, 4);

    // Channel assignment (4 bits)
    uint8_t ch_assign;
    if (nch == 2 && sm != 0) {
        ch_assign = (uint8_t)sm;        // 8=left+side, 9=right+side, 10=mid+side
    } else {
        ch_assign = (uint8_t)(nch - 1); // independent
    }
    bw.write_bits(ch_assign, 4);

    // Sample size / BPS (3 bits) + reserved (1 bit)
    bw.write_bits(encode_bps(bps), 3);
    bw.write_bits(0, 1);

    // Now byte-aligned: write UTF-8 coded sample number
    bw.write_utf8(sample_number);

    // Optional block-size trailing bytes
    if (bs_extra_bits > 0) bw.write_bits(bs_extra_val, bs_extra_bits);

    // Optional sample-rate trailing bytes
    if (sr_extra_bits > 0) bw.write_bits(sr_extra_val, sr_extra_bits);

    // CRC-8 of header (from sync up to, not including, CRC-8 byte)
    size_t hdr_end = bw.byte_size();
    bw.write_bits(bw.crc8(frame_start, hdr_end), 8);

    // --- Subframes ---------------------------------------------------------
    for (uint32_t c = 0; c < nch; ++c) {
        write_subframe(bw, params.subframes[c], ch[c].data(), bsize, bps_ch[c]);
    }

    // --- Frame footer ------------------------------------------------------
    bw.align();   // zero-pad to byte boundary
    size_t crc_end = bw.byte_size();
    uint16_t crc16 = bw.crc16(frame_start, crc_end);
    bw.write_bits(crc16 >> 8,   8);
    bw.write_bits(crc16 & 0xFF, 8);

    return bw.buffer();
}

// ============================================================
// Subframe serialization
// ============================================================

void FrameWriter::write_subframe(
    BitWriter&            bw,
    const SubframeParams& sp,
    const int32_t*        samples,
    uint32_t              bsize,
    uint32_t              ch_bps)
{
    const int    w    = sp.wasted_bits;
    const uint32_t eff = ch_bps - (uint32_t)w;   // effective bps for payload

    // ---- Subframe header: 0 | type[6] | wasted_bits_flag[+value] ---------
    // Zero padding bit
    bw.write_bits(0, 1);

    // 6-bit subframe type
    switch (sp.mode) {
        case 0: bw.write_bits(0x00, 6); break;                        // CONSTANT
        case 1: bw.write_bits(0x01, 6); break;                        // VERBATIM
        case 2: bw.write_bits(0x08 | sp.order, 6); break;             // FIXED
        case 3: bw.write_bits(0x20 | (sp.order - 1), 6); break;       // LPC
        default: break;
    }

    // Wasted-bits flag + unary-coded count
    if (w == 0) {
        bw.write_bits(0, 1);
    } else {
        bw.write_bits(1, 1);
        bw.write_unary((uint32_t)(w - 1));   // (k-1) zeros then a 1
    }

    // ---- Payload ----------------------------------------------------------
    if (sp.mode == 0) {
        // CONSTANT: one sample in eff bits (signed)
        bw.write_signed_bits(samples[0] >> w, (int)eff);
        return;
    }

    if (sp.mode == 1) {
        // VERBATIM: all samples
        for (uint32_t i = 0; i < bsize; ++i)
            bw.write_signed_bits(samples[i] >> w, (int)eff);
        return;
    }

    // FIXED and LPC share warm-up samples
    for (int i = 0; i < sp.order; ++i)
        bw.write_signed_bits(samples[i] >> w, (int)eff);

    std::vector<int32_t> residuals(bsize);

    if (sp.mode == 2) {
        // FIXED: compute residuals via finite differences
        for (uint32_t i = 0; i < bsize; ++i) {
            int32_t s  = samples[i]     >> w;
            int32_t s1 = (i > 0) ? (samples[i-1] >> w) : 0;
            int32_t s2 = (i > 1) ? (samples[i-2] >> w) : 0;
            int32_t s3 = (i > 2) ? (samples[i-3] >> w) : 0;
            int32_t s4 = (i > 3) ? (samples[i-4] >> w) : 0;
            if ((uint32_t)i < (uint32_t)sp.order) {
                residuals[i] = s;
            } else {
                switch (sp.order) {
                    case 0: residuals[i] = s; break;
                    case 1: residuals[i] = s - s1; break;
                    case 2: residuals[i] = s - 2*s1 + s2; break;
                    case 3: residuals[i] = s - 3*s1 + 3*s2 - s3; break;
                    case 4: residuals[i] = s - 4*s1 + 6*s2 - 4*s3 + s4; break;
                }
            }
        }
    } else {
        // LPC: write precision (4 bits) + shift (5-bit signed) + coefficients
        bw.write_bits((uint64_t)(sp.lpc_precision - 1), 4);
        bw.write_signed_bits(sp.lpc_shift, 5);
        for (int i = 0; i < sp.order; ++i)
            bw.write_signed_bits(sp.q_coeffs[i], sp.lpc_precision);

        // Compute residuals with the stored quantized coefficients
        for (uint32_t i = 0; i < bsize; ++i) {
            int32_t s = samples[i] >> w;
            if ((uint32_t)i < (uint32_t)sp.order) {
                residuals[i] = s;
            } else {
                int64_t pred = 0;
                for (int j = 0; j < sp.order; ++j)
                    pred += (int64_t)sp.q_coeffs[j] * (int64_t)(samples[i-1-j] >> w);
                residuals[i] = s - (int32_t)(pred >> sp.lpc_shift);
            }
        }
    }

    write_residual(bw, sp, residuals.data(), bsize, sp.order);
}

// ============================================================
// Rice-partitioned residual
// ============================================================

void FrameWriter::write_residual(
    BitWriter&            bw,
    const SubframeParams& sp,
    const int32_t*        residuals,
    uint32_t              bsize,
    int                   order)
{
    bw.write_bits(0, 2); // coding method = PARTITIONED_RICE (method 0)
    bw.write_bits((uint64_t)sp.rice_partition_order, 4);

    uint32_t num_parts = 1u << sp.rice_partition_order;
    uint32_t p_size    = bsize / num_parts;

    for (uint32_t p = 0; p < num_parts; ++p) {
        int      raw_k = sp.rice_k[p];
        uint32_t start = p * p_size;
        uint32_t end   = start + p_size;
        uint32_t first = std::max(start, (uint32_t)order); // skip warm-up in partition 0

        if (raw_k < 15) {
            // Normal Rice coding
            bw.write_bits((uint64_t)raw_k, 4);
            for (uint32_t i = first; i < end; ++i)
                bw.write_rice_sample(residuals[i], raw_k);
        } else {
            // Escape code (k = 15): verbatim residuals.
            // The escape_bps is stored in the upper bits of raw_k (see optimizer.cpp).
            int escape_bps = raw_k >> 8;
            if (escape_bps < 1)  escape_bps = 1;
            if (escape_bps > 32) escape_bps = 32;
            bw.write_bits(15u, 4);          // 4-bit escape marker
            bw.write_bits((uint64_t)escape_bps, 5); // 5-bit bits-per-sample
            for (uint32_t i = first; i < end; ++i)
                bw.write_signed_bits(residuals[i], escape_bps);
        }
    }
}

// ============================================================
// STREAMINFO block builder
// ============================================================

std::vector<uint8_t> FrameWriter::make_streaminfo_block(
    bool     is_last,
    uint32_t min_bs,  uint32_t max_bs,
    uint32_t min_frm, uint32_t max_frm,
    uint32_t sr, uint32_t channels, uint32_t bps,
    uint64_t total_samples,
    const uint8_t* md5)
{
    BitWriter bw;

    // 4-byte metadata block header
    bw.write_bits((uint64_t)(is_last ? 1 : 0), 1);
    bw.write_bits(0u, 7); // block type = STREAMINFO
    bw.write_bits(34u, 24); // STREAMINFO payload is always 34 bytes

    // 34-byte STREAMINFO payload
    bw.write_bits(min_bs,       16);
    bw.write_bits(max_bs,       16);
    bw.write_bits(min_frm,      24);
    bw.write_bits(max_frm,      24);
    bw.write_bits(sr,           20);
    bw.write_bits(channels - 1,  3);
    bw.write_bits(bps - 1,       5);
    bw.write_bits(total_samples, 36);

    // 16-byte MD5 signature (all-zeros = "not set" per spec)
    if (md5) {
        for (int i = 0; i < 16; ++i) bw.write_bits(md5[i], 8);
    } else {
        for (int i = 0; i < 16; ++i) bw.write_bits(0u, 8);
    }

    return bw.buffer(); // 38 bytes total
}
