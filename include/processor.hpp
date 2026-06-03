#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "FLAC/stream_decoder.h"

class Processor {
public:
    Processor(const std::string& input_file, const std::string& output_file);
    ~Processor();

    bool process();

private:
    // libflac decoder callbacks
    static FLAC__StreamDecoderWriteStatus write_callback(const FLAC__StreamDecoder *decoder, 
                                                         const FLAC__Frame *frame, 
                                                         const FLAC__int32 * const buffer[], 
                                                         void *client_data);
    static void error_callback(const FLAC__StreamDecoder *decoder, 
                              FLAC__StreamDecoderErrorStatus status, 
                              void *client_data);
    static void metadata_callback(const FLAC__StreamDecoder *decoder, 
                                 const FLAC__StreamMetadata *metadata, 
                                 void *client_data);

    std::string m_input;
    std::string m_output;
    
    // Internal PCM storage
    std::vector<std::vector<int32_t>> m_pcm_data; // per-channel
    uint32_t m_sample_rate = 0;
    uint32_t m_channels = 0;
    uint32_t m_bps = 0;
    uint64_t m_total_samples = 0;
};

#endif // PROCESSOR_HPP
