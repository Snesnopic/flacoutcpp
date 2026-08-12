#include "gpu_encoder.hpp"

#ifndef FLACOUT_HAVE_VULKAN

// Built without Vulkan. The stub keeps every call site compiling and makes `-P`
// a clear diagnostic rather than a link error, exactly as src/gpu.cpp does.
namespace flacoutcpp {
struct PureGpuEncoder::Impl {
    std::string why = "built without Vulkan support "
                      "(configure with -DFLACOUT_VULKAN=ON)";
};
PureGpuEncoder::PureGpuEncoder(uint32_t, uint32_t, uint32_t, const Config&)
    : m_impl(new Impl) {}
PureGpuEncoder::~PureGpuEncoder() = default;
bool PureGpuEncoder::available() const { return false; }
const std::string& PureGpuEncoder::why() const { return m_impl->why; }
bool PureGpuEncoder::encode(const std::vector<std::vector<int32_t>>&,
                            const Sink&, Stats*) { return false; }
} // namespace flacoutcpp

#else

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "frame_writer.hpp"

#include "pg_prepare_spv.h"
#include "pg_autoc_spv.h"
#include "pg_levinson_spv.h"
#include "pg_quant_spv.h"
#include "pg_fixed_spv.h"
#include "pg_sweep_spv.h"
#include "pg_select_spv.h"
#include "pg_rice_spv.h"
#include "pg_layout_spv.h"
#include "pg_frame_spv.h"
#include "pg_pack_spv.h"
#include "pg_crc_spv.h"

namespace flacoutcpp {

namespace {

// ---------------------------------------------------------------------------
// Layout constants shared with the shaders. Any change here is a change there.
// ---------------------------------------------------------------------------
constexpr int MAXO        = 32;   // LPC order ceiling
constexpr int NLAG        = 33;   // autocorrelation lags (0..32)
constexpr int CSTRIDE     = 36;   // ints per candidate record
constexpr int SFP_STRIDE  = 272;  // ints per subframe-params record
constexpr int PLAN_STRIDE = 8;    // ints per frame plan
constexpr int FINFO_STRIDE = 4;   // ints per frame layout record
constexpr int NFIXED      = 5;    // fixed predictors, orders 0..4
constexpr int MAXBIND     = 7;    // widest descriptor set (pg_pack)
constexpr int PUSH_MAX    = 64;   // bytes; covers every stage's block

enum Buf {
    B_PCM = 0, B_SIG, B_META, B_WIN, B_AUTOC, B_LPCF, B_CAND, B_COST,
    B_PLAN, B_RES, B_SFP, B_FINFO, B_TOTAL, B_OUT, B_ERR, B_COUNT
};

enum Stage {
    S_PREPARE = 0, S_AUTOC, S_LEVINSON, S_QUANT, S_FIXED, S_SWEEP,
    S_SELECT, S_RICE, S_LAYOUT, S_FRAME, S_PACK, S_CRC, S_COUNT
};

struct StageDef {
    const char*     name;
    const uint32_t* spv;
    size_t          spvBytes;
    int             nbind;
    int             bind[MAXBIND];
};

const StageDef kStages[S_COUNT] = {
    {"prepare",  kPgPrepareSpv,  sizeof(kPgPrepareSpv),  3, {B_PCM, B_SIG, B_META}},
    {"autoc",    kPgAutocSpv,    sizeof(kPgAutocSpv),    3, {B_SIG, B_WIN, B_AUTOC}},
    {"levinson", kPgLevinsonSpv, sizeof(kPgLevinsonSpv), 3, {B_AUTOC, B_LPCF, B_ERR}},
    {"quant",    kPgQuantSpv,    sizeof(kPgQuantSpv),    4, {B_LPCF, B_META, B_CAND, B_ERR}},
    {"fixed",    kPgFixedSpv,    sizeof(kPgFixedSpv),    2, {B_META, B_CAND}},
    {"sweep",    kPgSweepSpv,    sizeof(kPgSweepSpv),    3, {B_SIG, B_CAND, B_COST}},
    {"select",   kPgSelectSpv,   sizeof(kPgSelectSpv),   4, {B_CAND, B_COST, B_META, B_PLAN}},
    {"rice",     kPgRiceSpv,     sizeof(kPgRiceSpv),     6, {B_SIG, B_CAND, B_PLAN, B_META, B_RES, B_SFP}},
    {"layout",   kPgLayoutSpv,   sizeof(kPgLayoutSpv),   3, {B_SFP, B_FINFO, B_TOTAL}},
    {"frame",    kPgFrameSpv,    sizeof(kPgFrameSpv),    3, {B_PLAN, B_FINFO, B_OUT}},
    {"pack",     kPgPackSpv,     sizeof(kPgPackSpv),     7, {B_SIG, B_RES, B_SFP, B_CAND, B_PLAN, B_FINFO, B_OUT}},
    {"crc",      kPgCrcSpv,      sizeof(kPgCrcSpv),      2, {B_FINFO, B_OUT}},
};

struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    void*          map  = nullptr;
    VkDeviceSize   size = 0;
};

// FLAC's 4-bit block-size code, plus the trailing field for sizes it cannot
// name directly. Replicated from frame_writer.cpp (static there) because the
// frame-header kernel needs the same numbers as push constants.
uint8_t blocksizeCode(uint32_t bs, uint32_t& extraVal, int& extraBits) {
    extraVal = 0; extraBits = 0;
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
        default: break;
    }
    if (bs <= 256) { extraVal = bs - 1; extraBits = 8;  return 0x6; }
    extraVal = bs - 1;  extraBits = 16; return 0x7;
}

// FLAC's 3-bit sample-size code. Replicated rather than reached for through
// FrameWriter, where it is private.
uint8_t bpsCodeOf(uint32_t bps) {
    switch (bps) {
        case 8:  return 0x1;
        case 12: return 0x2;
        case 16: return 0x4;
        case 20: return 0x5;
        case 24: return 0x6;
        case 32: return 0x7;
        default: return 0x0;   // from STREAMINFO
    }
}

uint8_t samplerateCode(uint32_t sr, uint32_t& extraVal, int& extraBits) {
    extraVal = 0; extraBits = 0;
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
        default: break;
    }
    if (sr % 1000 == 0 && sr / 1000 <= 255)   { extraVal = sr / 1000; extraBits = 8;  return 0xC; }
    if (sr <= 65535)                          { extraVal = sr;        extraBits = 16; return 0xD; }
    if (sr % 10 == 0 && sr / 10 <= 65535)     { extraVal = sr / 10;   extraBits = 16; return 0xE; }
    return 0x0;   // fall back to STREAMINFO
}

} // namespace

// ---------------------------------------------------------------------------

struct PureGpuEncoder::Impl {
    std::string why;
    bool ok = false;

    uint32_t nch, bps, srate;
    Config   cfg;

    // Derived shape
    uint32_t B      = 4096;
    int      nsig   = 1;      // 1 mono, 4 stereo (L, R, M, S)
    int      nsub   = 1;      // subframes per frame == channels
    int      nwin   = 1;
    int      nprec  = 1;
    int      precs[4]{15, 0, 0, 0};
    int      ncand  = 0;      // slots per (signal, block)
    int      lpcSlots = 0;
    /// Orders swept per (block, signal, window), chosen by the Levinson-error
    /// ranking in pg_quant. The rest are skipped before the sweep touches them.
    int      norders = 32;
    uint32_t maxBlk = 256;
    /// Ridge used by pg_levinson's *retry* path only -- a solve that completes
    /// cleanly never sees it, which is why real music is byte-identical with the
    /// retry on or off. 1e-6 measured best of {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3}
    /// (see the table in pg_levinson.comp). FLACOUT_PG_RIDGE overrides it.
    float    ridge  = 1e-6f;
    /// FLACOUT_PG_PROFILE: fence-wait after every stage and accumulate its cost.
    bool     profile = false;
    double   stage_secs[S_COUNT]{};

    VkInstance       inst  = VK_NULL_HANDLE;
    VkPhysicalDevice phys  = VK_NULL_HANDLE;
    VkDevice         dev   = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         qfam  = 0;
    VkPhysicalDeviceMemoryProperties memprops{};
    bool size_ctl = false;

    VkDescriptorSetLayout dsl  = VK_NULL_HANDLE;
    VkDescriptorPool      dpool = VK_NULL_HANDLE;
    VkPipelineLayout      plo  = VK_NULL_HANDLE;
    VkCommandPool         cpool = VK_NULL_HANDLE;
    VkCommandBuffer       cmd  = VK_NULL_HANDLE;
    VkFence               fence = VK_NULL_HANDLE;

    VkShaderModule  mods[S_COUNT]{};
    VkPipeline      pipes[S_COUNT]{};
    VkDescriptorSet dsets[S_COUNT]{};

    Buffer bufs[B_COUNT];

    bool init();
    void destroy();
    uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags want) const;
    bool alloc(Buffer& b, VkDeviceSize need);
    bool allocAll();
    void bindSets();
    void uploadWindows();
    bool runChunk(const std::vector<std::vector<int32_t>>& pcm,
                  uint64_t firstSample, uint32_t nblkThis,
                  uint32_t* outBytes);
};

uint32_t PureGpuEncoder::Impl::findMem(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

bool PureGpuEncoder::Impl::alloc(Buffer& b, VkDeviceSize need) {
    if (need == 0) need = 256;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = need;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;   // vkCmdFillBuffer on B_OUT
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    // Host-visible for every buffer: the only transfers are the PCM upload and
    // the encoded-bytes readback, and on unified memory both are free. On a
    // discrete part this trades a slower device-side access for not having to
    // stage, which is the right trade until a profile says otherwise.
    const uint32_t idx = findMem(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (idx == UINT32_MAX) return false;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = idx;
    if (vkAllocateMemory(dev, &ai, nullptr, &b.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(dev, b.buf, b.mem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map) != VK_SUCCESS) return false;
    b.size = need;
    return true;
}

bool PureGpuEncoder::Impl::allocAll() {
    const uint64_t nb   = maxBlk;
    const uint64_t sigN = (uint64_t)nsig * nb * B;
    const uint64_t solv = (uint64_t)nsig * nb * nwin;
    const uint64_t cndN = (uint64_t)nsig * nb * ncand;

    // Worst-case encoded size: every subframe VERBATIM at bps+1, plus generous
    // per-frame overhead. VERBATIM is a hard ceiling -- the search can only pick
    // it when nothing else is smaller -- so the output buffer cannot overflow.
    const uint64_t outN = nb * ((uint64_t)B * nsub * ((bps + 1 + 7) / 8) + 64);

    struct { int id; uint64_t bytes; } want[] = {
        {B_PCM,   (uint64_t)nch * nb * B * 4},
        {B_SIG,   sigN * 4},
        {B_META,  (uint64_t)nsig * nb * 4},
        {B_WIN,   (uint64_t)nwin * B * 4},
        {B_AUTOC, solv * NLAG * 4},
        {B_LPCF,  solv * MAXO * MAXO * 4},
        {B_CAND,  cndN * CSTRIDE * 4},
        {B_COST,  cndN * 4},
        {B_PLAN,  nb * PLAN_STRIDE * 4},
        {B_RES,   nb * nsub * B * 4},
        {B_SFP,   nb * nsub * SFP_STRIDE * 4},
        {B_FINFO, nb * FINFO_STRIDE * 4},
        {B_TOTAL, 16},
        {B_ERR,   solv * NLAG * 4},
        {B_OUT,   outN},
    };
    uint64_t total = 0;
    for (auto& w : want) {
        if (!alloc(bufs[w.id], w.bytes)) { why = "device buffer allocation failed"; return false; }
        total += w.bytes;
    }
    if (cfg.verbose)
        std::cout << "Pure GPU: " << (total >> 20) << " MB device buffers for "
                  << maxBlk << "-frame chunks\n";
    return true;
}

void PureGpuEncoder::Impl::bindSets() {
    for (int s = 0; s < S_COUNT; ++s) {
        VkDescriptorBufferInfo dbi[MAXBIND]{};
        VkWriteDescriptorSet   wr[MAXBIND]{};
        for (int i = 0; i < MAXBIND; ++i) {
            // Bindings past the shader's own count are bound to something valid
            // anyway: the descriptor set layout is shared across stages, and an
            // unbound-but-declared binding is undefined behaviour even when the
            // shader never reads it.
            const int b = (i < kStages[s].nbind) ? kStages[s].bind[i] : B_TOTAL;
            dbi[i].buffer = bufs[b].buf;
            dbi[i].range  = VK_WHOLE_SIZE;
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = dsets[s];
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, MAXBIND, wr, 0, nullptr);
    }
}

void PureGpuEncoder::Impl::uploadWindows() {
    std::vector<double> tmp(B);
    float* dst = (float*)bufs[B_WIN].map;
    for (int w = 0; w < nwin; ++w) {
        window_coefficients(cfg.windows[w], B, tmp.data());
        for (uint32_t i = 0; i < B; ++i) dst[w * B + i] = (float)tmp[i];
    }
}

bool PureGpuEncoder::Impl::init() {
    // ---- shape -----------------------------------------------------------
    B = cfg.block_size;
    if (B % 256 != 0 || B > 4096 || B < 256) {
        why = "block size must be a multiple of 256 in [256, 4096]";
        return false;
    }
    if (nch < 1 || nch > 2) { why = "pure-GPU path supports 1 or 2 channels"; return false; }
    nsub = (int)nch;
    nsig = (nch == 2) ? 4 : 1;
    nwin = (int)cfg.windows.size();
    if (nwin < 1 || nwin > 16) { why = "window count must be 1..16"; return false; }
    nprec = (int)cfg.precisions.size();
    if (nprec < 1 || nprec > 4) { why = "precision count must be 1..4"; return false; }
    for (int i = 0; i < nprec; ++i) {
        precs[i] = cfg.precisions[i];
        if (precs[i] < 5 || precs[i] > 15) { why = "precisions must be 5..15"; return false; }
    }
    lpcSlots = nwin * MAXO * nprec;
    ncand    = lpcSlots + NFIXED;
    maxBlk   = std::max(1u, std::min(cfg.blocks_per_chunk, 1024u));
    if (const char* r = std::getenv("FLACOUT_PG_RIDGE")) ridge = (float)atof(r);
    norders = (int)std::max(1u, std::min(cfg.orders, 32u));
    profile = std::getenv("FLACOUT_PG_PROFILE") != nullptr;

    // ---- instance --------------------------------------------------------
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
    std::vector<VkExtensionProperties> ie(nie);
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, ie.data());
    std::vector<const char*> iexts;
    VkInstanceCreateFlags iflags = 0;
    for (const auto& e : ie) {
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_enumeration")) {
            iexts.push_back("VK_KHR_portability_enumeration");
            iflags |= 0x00000001u;
        } else if (!std::strcmp(e.extensionName,
                                "VK_KHR_get_physical_device_properties2")) {
            iexts.push_back("VK_KHR_get_physical_device_properties2");
        }
    }
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "flacoutcpp";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.flags = iflags;
    ici.enabledExtensionCount = (uint32_t)iexts.size();
    ici.ppEnabledExtensionNames = iexts.data();
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        why = "no Vulkan loader or instance creation failed"; return false;
    }

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    if (!nd) { why = "no Vulkan devices"; return false; }
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(inst, &nd, devs.data());
    int pick = 0;
    for (uint32_t i = 0; i < nd; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = (int)i; break; }
    }
    phys = devs[pick];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    uint32_t nde = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, nullptr);
    std::vector<VkExtensionProperties> de(nde);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, de.data());
    for (const auto& e : de)
        if (!std::strcmp(e.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            size_ctl = true;

    VkPhysicalDeviceSubgroupProperties sgp{};
    sgp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgc{};
    sgc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sgp;
    if (size_ctl) sgp.pNext = &sgc;
    vkGetPhysicalDeviceProperties2(phys, &p2);

    if (size_ctl) {
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgf{};
        sgf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &sgf;
        vkGetPhysicalDeviceFeatures2(phys, &f2);
        if (!sgf.subgroupSizeControl || !sgf.computeFullSubgroups ||
            !(sgc.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            sgc.minSubgroupSize > 32u || sgc.maxSubgroupSize < 32u)
            size_ctl = false;
    }
    // Three kernels map one bit-plane (or one LPC tap) per lane, so 32 is not a
    // tuning preference. GPU_PLAN.md records what happens when this is assumed
    // rather than pinned: the kernel's own width guard fires at runtime and its
    // error sentinel wins the search.
    if (!size_ctl && sgp.subgroupSize != 32) {
        why = std::string(props.deviceName) + ": subgroup size is " +
              std::to_string(sgp.subgroupSize) +
              ", these kernels require 32 (and VK_EXT_subgroup_size_control is "
              "unavailable to pin it)";
        return false;
    }
    const VkSubgroupFeatureFlags needsg =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    if ((sgp.supportedOperations & needsg) != needsg) {
        why = std::string(props.deviceName) +
              ": missing required subgroup operations "
              "(ballot/arithmetic/shuffle/shuffle-relative)";
        return false;
    }
    VkPhysicalDeviceFeatures feat{};
    vkGetPhysicalDeviceFeatures(phys, &feat);
    if (!feat.shaderInt64) {
        why = std::string(props.deviceName) + ": shaderInt64 unavailable";
        return false;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp.data());
    bool found = false;
    for (uint32_t i = 0; i < nq; ++i)
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qfam = i; found = true; break; }
    if (!found)
        for (uint32_t i = 0; i < nq; ++i)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; found = true; break; }
    if (!found) { why = "no compute queue"; return false; }

    std::vector<const char*> dexts;
    for (const auto& e : de)
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset"))
            dexts.push_back("VK_KHR_portability_subset");
    if (size_ctl) dexts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures want{};
    want.shaderInt64 = VK_TRUE;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgf_on{};
    sgf_on.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    sgf_on.subgroupSizeControl  = VK_TRUE;
    sgf_on.computeFullSubgroups = VK_TRUE;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (size_ctl) dci.pNext = &sgf_on;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dexts.size();
    dci.ppEnabledExtensionNames = dexts.data();
    dci.pEnabledFeatures = &want;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        why = std::string(props.deviceName) + ": device creation failed"; return false;
    }
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // ---- descriptors and pipelines ---------------------------------------
    VkDescriptorSetLayoutBinding bind[MAXBIND]{};
    for (int i = 0; i < MAXBIND; ++i) {
        bind[i].binding = (uint32_t)i;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = MAXBIND;
    dslci.pBindings = bind;
    if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl) != VK_SUCCESS) {
        why = "descriptor set layout"; return false;
    }
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAXBIND * S_COUNT};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = S_COUNT;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool) != VK_SUCCESS) {
        why = "descriptor pool"; return false;
    }
    for (int s = 0; s < S_COUNT; ++s) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(dev, &dsai, &dsets[s]) != VK_SUCCESS) {
            why = "descriptor set"; return false;
        }
    }

    // One pipeline layout for every stage: a push-constant range wide enough for
    // the largest block, and a shader declaring a smaller one is compatible.
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_MAX};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &plo) != VK_SUCCESS) {
        why = "pipeline layout"; return false;
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss{};
    rss.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    rss.requiredSubgroupSize = 32;

    for (int s = 0; s < S_COUNT; ++s) {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = kStages[s].spvBytes;
        smci.pCode    = kStages[s].spv;
        if (vkCreateShaderModule(dev, &smci, nullptr, &mods[s]) != VK_SUCCESS) {
            why = std::string("shader module: ") + kStages[s].name; return false;
        }
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mods[s];
        cpi.stage.pName = "main";
        if (size_ctl) {
            cpi.stage.pNext = &rss;
            cpi.stage.flags =
                VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
        }
        cpi.layout = plo;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                     &pipes[s]) != VK_SUCCESS) {
            why = std::string(props.deviceName) + ": pipeline creation failed for "
                + kStages[s].name;
            return false;
        }
    }

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qfam;
    if (vkCreateCommandPool(dev, &cpci, nullptr, &cpool) != VK_SUCCESS) {
        why = "command pool"; return false;
    }
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev, &cbai, &cmd) != VK_SUCCESS) {
        why = "command buffer"; return false;
    }
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS) {
        why = "fence"; return false;
    }

    if (!allocAll()) return false;
    bindSets();
    uploadWindows();

    why = std::string(props.deviceName) + " (subgroup 32 " +
          (size_ctl ? "pinned" : "by default") + ", shaderInt64, fp32 analysis)";
    ok = true;
    return true;
}

void PureGpuEncoder::Impl::destroy() {
    if (dev == VK_NULL_HANDLE) { if (inst) vkDestroyInstance(inst, nullptr); return; }
    vkDeviceWaitIdle(dev);
    for (int i = 0; i < B_COUNT; ++i) {
        if (bufs[i].map) vkUnmapMemory(dev, bufs[i].mem);
        if (bufs[i].buf) vkDestroyBuffer(dev, bufs[i].buf, nullptr);
        if (bufs[i].mem) vkFreeMemory(dev, bufs[i].mem, nullptr);
    }
    for (int s = 0; s < S_COUNT; ++s) {
        if (pipes[s]) vkDestroyPipeline(dev, pipes[s], nullptr);
        if (mods[s])  vkDestroyShaderModule(dev, mods[s], nullptr);
    }
    if (fence) vkDestroyFence(dev, fence, nullptr);
    if (cpool) vkDestroyCommandPool(dev, cpool, nullptr);
    if (plo)   vkDestroyPipelineLayout(dev, plo, nullptr);
    if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
    if (dsl)   vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (inst)  vkDestroyInstance(inst, nullptr);
}

// ---------------------------------------------------------------------------

namespace {

void barrier(VkCommandBuffer cmd) {
    // One global barrier between stages. Every stage reads what the previous one
    // wrote, so there is nothing to overlap and nothing subtler to express.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

inline uint32_t ceilDiv(uint64_t a, uint64_t b) { return (uint32_t)((a + b - 1) / b); }

} // namespace

bool PureGpuEncoder::Impl::runChunk(
    const std::vector<std::vector<int32_t>>& pcm,
    uint64_t firstSample, uint32_t nblkThis, uint32_t* outBytes)
{
    const uint32_t chunkLen = nblkThis * B;

    // ---- upload PCM (planar) ---------------------------------------------
    {
        int32_t* dst = (int32_t*)bufs[B_PCM].map;
        for (uint32_t c = 0; c < nch; ++c)
            std::memcpy(dst + (size_t)c * chunkLen,
                        pcm[c].data() + firstSample,
                        (size_t)chunkLen * sizeof(int32_t));
    }

    uint32_t bsExtraVal, srExtraVal; int bsExtraBits, srExtraBits;
    const uint8_t bsCode = blocksizeCode(B, bsExtraVal, bsExtraBits);
    const uint8_t srCode = samplerateCode(srate, srExtraVal, srExtraBits);
    const uint8_t bpsCode = bpsCodeOf(bps);

    const uint64_t nsolve = (uint64_t)nblkThis * nsig * nwin;
    const uint64_t ncTot  = (uint64_t)nblkThis * nsig * ncand;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);

    // The packer ORs bits into this buffer, so it has to start at zero -- which
    // is also what makes a unary run's zeros free.
    vkCmdFillBuffer(cmd, bufs[B_OUT].buf, 0, VK_WHOLE_SIZE, 0);
    barrier(cmd);

    // FLACOUT_PG_PROFILE: submit each stage on its own and fence-wait, so the
    // per-stage cost is visible. It serialises the queue and adds a submit per
    // stage, so the total is an overestimate -- read the *shares*, not the sum,
    // and never quote a speed number from a profiling run (CLAUDE.md trap 4).
    auto go = [&](int stage, const void* push, size_t pushSize,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[stage]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1,
                                &dsets[stage], 0, nullptr);
        vkCmdPushConstants(cmd, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           (uint32_t)pushSize, push);
        vkCmdDispatch(cmd, gx, gy, gz);
        barrier(cmd);
        if (!profile) return;
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        const auto t0 = std::chrono::steady_clock::now();
        vkResetFences(dev, 1, &fence);
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
        stage_secs[stage] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        VkCommandBufferBeginInfo rb{};
        rb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        rb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &rb);
    };

    { struct { int32_t nblk, bsize, nch, chunkLen; } p{
        (int32_t)nblkThis, (int32_t)B, (int32_t)nch, (int32_t)chunkLen };
      go(S_PREPARE, &p, sizeof p, nblkThis, (uint32_t)nsig); }

    { struct { int32_t nblk, bsize, nwin; float scale; } p{
        (int32_t)nblkThis, (int32_t)B, nwin, 1.0f / (float)(1u << (bps - 1)) };
      go(S_AUTOC, &p, sizeof p, nblkThis, (uint32_t)nsig, (uint32_t)nwin); }

    { struct { int32_t nsolve; float ridge; } p{ (int32_t)nsolve, ridge };
      go(S_LEVINSON, &p, sizeof p, ceilDiv(nsolve, 4)); }   // 128 lanes / 32

    { struct { int32_t nblk, bsize, nsig, nwin, nprec, ncand, bps,
               p0, p1, p2, p3, norders; } p{
        (int32_t)nblkThis, (int32_t)B, nsig, nwin, nprec, ncand, (int32_t)bps,
        precs[0], precs[1], precs[2], precs[3], norders };
      go(S_QUANT, &p, sizeof p, ceilDiv(nsolve * MAXO * nprec, 64)); }

    { struct { int32_t nblk, bsize, nsig, ncand, lpcSlots, bps; } p{
        (int32_t)nblkThis, (int32_t)B, nsig, ncand, lpcSlots, (int32_t)bps };
      go(S_FIXED, &p, sizeof p, ceilDiv((uint64_t)nblkThis * nsig * NFIXED, 64)); }

    { struct { int32_t ncand, bsize, maxPOrder; } p{
        (int32_t)ncTot, (int32_t)B, 8 };
      go(S_SWEEP, &p, sizeof p, ceilDiv(ncTot, 4)); }

    { struct { int32_t nblk, bsize, nsig, ncand, bps; } p{
        (int32_t)nblkThis, (int32_t)B, nsig, ncand, (int32_t)bps };
      go(S_SELECT, &p, sizeof p, nblkThis); }

    { struct { int32_t nblk, bsize, nsig, nsub, ncand, nwin, nprec, bps,
               p0, p1, p2, p3; } p{
        (int32_t)nblkThis, (int32_t)B, nsig, nsub, ncand, nwin, nprec, (int32_t)bps,
        precs[0], precs[1], precs[2], precs[3] };
      go(S_RICE, &p, sizeof p, nblkThis * (uint32_t)nsub); }

    { struct { int32_t nframe, bsize, nsub, bsExtraBits, srExtraBits;
               uint32_t baseLo, baseHi; } p{
        (int32_t)nblkThis, (int32_t)B, nsub, bsExtraBits, srExtraBits,
        (uint32_t)(firstSample & 0xFFFFFFFFu), (uint32_t)(firstSample >> 32) };
      go(S_LAYOUT, &p, sizeof p, 1); }

    { struct { int32_t nframe, bsize, nch, bsCode, srCode, bsExtraVal,
               bsExtraBits, srExtraVal, srExtraBits, bpsCode;
               uint32_t baseLo, baseHi; } p{
        (int32_t)nblkThis, (int32_t)B, (int32_t)nch, bsCode, srCode,
        (int32_t)bsExtraVal, bsExtraBits, (int32_t)srExtraVal, srExtraBits,
        bpsCode,
        (uint32_t)(firstSample & 0xFFFFFFFFu), (uint32_t)(firstSample >> 32) };
      go(S_FRAME, &p, sizeof p, ceilDiv(nblkThis, 64)); }

    { struct { int32_t nblk, bsize, nsub, nsig, ncand; } p{
        (int32_t)nblkThis, (int32_t)B, nsub, nsig, ncand };
      go(S_PACK, &p, sizeof p, nblkThis * (uint32_t)nsub); }

    { struct { int32_t nframe; } p{ (int32_t)nblkThis };
      go(S_CRC, &p, sizeof p, ceilDiv(nblkThis, 256)); }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(dev, 1, &fence);
    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) return false;
    if (vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;

    *outBytes = ((const uint32_t*)bufs[B_TOTAL].map)[0];
    if ((VkDeviceSize)*outBytes > bufs[B_OUT].size) return false;

    // FLACOUT_PG_DEBUG: dump the first solve's analysis chain next to a
    // double-precision host reference. This exists to split "the analysis is
    // numerically wrong" from "the analysis is fine and something downstream
    // mis-prices it", which is the one distinction the output size cannot make.
    if (std::getenv("FLACOUT_PG_DEBUG") && firstSample == 0) {
        const float* ac = (const float*)bufs[B_AUTOC].map;
        const float* lp = (const float*)bufs[B_LPCF].map;
        const int32_t* sg = (const int32_t*)bufs[B_SIG].map;
        const uint32_t* mt = (const uint32_t*)bufs[B_META].map;

        std::vector<double> wc(B), wd(B);
        window_coefficients(cfg.windows[0], B, wc.data());
        const double scale = 1.0 / (double)(1u << (bps - 1));
        for (uint32_t i = 0; i < B; ++i) wd[i] = (double)sg[i] * scale * wc[i];

        std::cout << "PG_DEBUG solve0 (blk 0, sig 0, win 0) meta=" << mt[0] << "\n";
        std::cout << "  lag   device(fp32)         host(double)        rel.err\n";
        for (int l = 0; l <= 8; ++l) {
            double ref = 0.0;
            for (uint32_t i = 0; i + (uint32_t)l < B; ++i) ref += wd[i] * wd[i + l];
            const double got = ac[l];
            const double rel = (ref != 0.0) ? (got - ref) / ref : 0.0;
            std::printf("  %3d  %18.8g  %18.8g  %12.3g\n", l, got, ref, rel);
        }
        std::cout << "  order-8 device coefficients:";
        for (int j = 0; j < 8; ++j) std::printf(" %.6f", lp[7 * MAXO + j]);
        std::cout << "\n  order-32 device coefficients (first 8):";
        for (int j = 0; j < 8; ++j) std::printf(" %.6f", lp[31 * MAXO + j]);
        std::cout << "\n";

        const int32_t* cd = (const int32_t*)bufs[B_CAND].map;
        const uint32_t* ct = (const uint32_t*)bufs[B_COST].map;
        const int32_t* pl = (const int32_t*)bufs[B_PLAN].map;
        std::cout << "  plan[0]: mode=" << pl[0] << " sig=" << pl[1] << "," << pl[2]
                  << " slot=" << pl[3] << "," << pl[4]
                  << " bits=" << pl[5] << "," << pl[6] << "\n";

        for (int s = 0; s < nsig; ++s) {
            const int bs = s * (int)nblkThis + 0;
            uint32_t best = UINT32_MAX; int bslot = -1;
            for (int k = 0; k < ncand; ++k) {
                const uint32_t rc = ct[bs * ncand + k];
                if (rc == UINT32_MAX) continue;
                const uint32_t tot = (uint32_t)cd[(bs * ncand + k) * CSTRIDE + 3] + rc;
                if (tot < best) { best = tot; bslot = k; }
            }
            int mn = INT32_MAX, mx = INT32_MIN;
            for (uint32_t i = 0; i < B; ++i) {
                mn = std::min(mn, sg[(size_t)bs * B + i]);
                mx = std::max(mx, sg[(size_t)bs * B + i]);
            }
            std::printf("  sig %d: meta=%u range=[%d,%d] best slot=%d bits=%u\n",
                        s, mt[bs], mn, mx, bslot, best);
            for (int ord : {1, 2, 4, 7, 8, 16, 32}) {
                const int ci = bs * ncand + (ord - 1) * nprec;
                std::printf("      ord=%2d shift=%2d hdr=%6d rice=%10u total=%10u\n",
                            cd[ci * CSTRIDE + 0], cd[ci * CSTRIDE + 1],
                            cd[ci * CSTRIDE + 3], ct[ci],
                            (uint32_t)cd[ci * CSTRIDE + 3] + ct[ci]);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------

PureGpuEncoder::PureGpuEncoder(uint32_t channels, uint32_t bps,
                               uint32_t sample_rate, const Config& cfg)
    : m_impl(new Impl)
{
    m_impl->nch   = channels;
    m_impl->bps   = bps;
    m_impl->srate = sample_rate;
    m_impl->cfg   = cfg;
    if (m_impl->cfg.windows.empty())
        m_impl->cfg.windows = {WindowType::TUKEY_050, WindowType::HANN,
                               WindowType::WELCH, WindowType::RECTANGULAR};
    if (!m_impl->init()) m_impl->ok = false;
}

PureGpuEncoder::~PureGpuEncoder() { m_impl->destroy(); }

bool PureGpuEncoder::available() const { return m_impl->ok; }
const std::string& PureGpuEncoder::why() const { return m_impl->why; }

bool PureGpuEncoder::encode(const std::vector<std::vector<int32_t>>& pcm,
                            const Sink& sink, Stats* out)
{
    Impl& I = *m_impl;
    if (!I.ok || pcm.empty()) return false;

    Stats st{};
    st.min_frame = UINT32_MAX;
    st.min_block = UINT32_MAX;

    const uint64_t total = pcm[0].size();
    const uint64_t nfull = total / I.B;
    const uint64_t tail  = total % I.B;

    const auto t0 = std::chrono::steady_clock::now();

    for (uint64_t done = 0; done < nfull; ) {
        const uint32_t n = (uint32_t)std::min<uint64_t>(I.maxBlk, nfull - done);
        uint32_t bytes = 0;
        if (!I.runChunk(pcm, done * I.B, n, &bytes)) return false;

        const uint32_t* fi = (const uint32_t*)I.bufs[B_FINFO].map;
        for (uint32_t f = 0; f < n; ++f) {
            const uint32_t fb = fi[f * FINFO_STRIDE + 1];
            st.min_frame = std::min(st.min_frame, fb);
            st.max_frame = std::max(st.max_frame, fb);
        }
        st.min_block = std::min(st.min_block, I.B);
        st.max_block = std::max(st.max_block, I.B);
        st.frames += n;
        st.bytes  += bytes;

        if (!sink((const uint8_t*)I.bufs[B_OUT].map, bytes)) return false;
        done += n;
    }

    // The stream tail -- fewer than block_size samples -- is emitted as one
    // VERBATIM frame on the host. It is the only frame the host builds, and the
    // cost is bounded by one block of uncompressed audio (~16 KB at 16/4096),
    // which is why it has not been worth a second set of kernels for a
    // non-power-of-two block size. Losslessness does not care.
    if (tail > 0) {
        BlockParams bp{};
        bp.block_size  = (uint32_t)tail;
        bp.stereo_mode = 0;
        for (uint32_t c = 0; c < I.nch; ++c) {
            bp.subframes[c].mode  = 1;      // VERBATIM
            bp.subframes[c].order = 0;
            bp.subframes[c].wasted_bits = 0;
        }
        FrameWriter fw;
        auto fb = fw.write_frame(bp, pcm, nfull * I.B, I.srate, I.bps);
        if (fb.empty()) return false;
        st.min_frame = std::min(st.min_frame, (uint32_t)fb.size());
        st.max_frame = std::max(st.max_frame, (uint32_t)fb.size());
        st.min_block = std::min(st.min_block, (uint32_t)tail);
        st.max_block = std::max(st.max_block, (uint32_t)tail);
        st.frames += 1;
        st.bytes  += fb.size();
        if (!sink(fb.data(), fb.size())) return false;
    }

    st.device_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (I.profile) {
        double tot = 0.0;
        for (int s = 0; s < S_COUNT; ++s) tot += I.stage_secs[s];
        std::cout << "PG_PROFILE (serialised submits; read shares, not the sum)\n";
        for (int s = 0; s < S_COUNT; ++s)
            std::printf("  %-9s %8.4f s  %5.1f%%\n", kStages[s].name,
                        I.stage_secs[s], tot > 0 ? 100.0 * I.stage_secs[s] / tot : 0.0);
        std::printf("  %-9s %8.4f s\n", "total", tot);
    }
    if (st.min_frame == UINT32_MAX) st.min_frame = 0;
    if (st.min_block == UINT32_MAX) st.min_block = 0;
    if (out) *out = st;
    return true;
}

} // namespace flacoutcpp

#endif // FLACOUT_HAVE_VULKAN
