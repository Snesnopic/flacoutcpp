#include "gpu.hpp"

#ifndef FLACOUT_HAVE_VULKAN

// Built without Vulkan. The stub keeps every call site compiling unchanged and
// makes `-G` a clear diagnostic instead of a link error.
namespace flacoutcpp {
struct GpuEvaluator::Impl { std::string why = "built without Vulkan support "
    "(configure with -DFLACOUT_VULKAN=ON)"; };
GpuEvaluator::GpuEvaluator() : m_impl(new Impl) {}
GpuEvaluator::~GpuEvaluator() = default;
bool GpuEvaluator::available() const { return false; }
const std::string& GpuEvaluator::why() const { return m_impl->why; }
bool GpuEvaluator::evaluate(const int32_t*, uint32_t,
                            const std::vector<Candidate>&,
                            std::vector<uint32_t>&) { return false; }
void GpuEvaluator::stats(uint64_t* c, double* s) const { if(c)*c=0; if(s)*s=0.0; }
void GpuEvaluator::set_min_batch(size_t) {}
size_t GpuEvaluator::min_batch() const { return 0; }
void GpuEvaluator::set_partition_cap(int) {}
int GpuEvaluator::partition_cap() const { return 8; }
void GpuEvaluator::set_slots(int) {}
int GpuEvaluator::slots() const { return 0; }
void GpuEvaluator::set_duty(int) {}
int GpuEvaluator::duty() const { return 100; }
bool GpuEvaluator::would_accept() const { return false; }
uint64_t GpuEvaluator::macs() const { return 0; }
} // namespace flacoutcpp

#else

#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>

#include "sweep_spv.h"   // generated at build time from shaders/sweep.comp

namespace flacoutcpp {

namespace {

// Growable device-local, host-visible buffer. Apple silicon and every
// integrated part expose one such heap, and on discrete parts this still works
// (BAR-visible memory) at the cost of PCIe reads — which is acceptable because
// the transfers here are tiny next to the compute.
struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    void*          map  = nullptr;
    VkDeviceSize   size = 0;
};

struct PushConsts {
    int32_t ncand;
    int32_t bsize;
    int32_t flags;
    int32_t maxPOrder;
};

} // namespace

struct GpuEvaluator::Impl {
    std::string why;
    bool ok = false;

    VkInstance       inst   = VK_NULL_HANDLE;
    VkPhysicalDevice phys   = VK_NULL_HANDLE;
    VkDevice         dev    = VK_NULL_HANDLE;
    VkQueue          queue  = VK_NULL_HANDLE;
    uint32_t         qfam   = 0;
    VkPhysicalDeviceMemoryProperties memprops{};

    VkDescriptorSetLayout dsl   = VK_NULL_HANDLE;
    VkDescriptorPool      pool  = VK_NULL_HANDLE;
    VkPipelineLayout      plo   = VK_NULL_HANDLE;
    VkPipeline            pipe  = VK_NULL_HANDLE;
    VkShaderModule        shader= VK_NULL_HANDLE;
    // Independent slots, so several workers can have dispatches in flight.
    // With a single command buffer the whole submit-and-wait sat inside one
    // lock, which capped the device at one dispatch at a time: measured, the
    // GPU absorbed only ~31% of the search while 15 of 16 threads took the CPU
    // path. Only vkQueueSubmit needs serialising (the queue is externally
    // synchronised); waiting is per-slot and lock-free.
    static constexpr int NSLOT = 16;   // allocated; `nslot` is how many are used
    struct Slot {
        // One pool per slot. A VkCommandPool is externally synchronised, so
        // two threads may not record into buffers from the same pool at once;
        // sharing one pool across slots is a data race that happens to survive
        // at low slot counts and produced differing output at 16.
        VkCommandPool   pool  = VK_NULL_HANDLE;
        VkCommandBuffer cmd   = VK_NULL_HANDLE;
        VkFence         fence = VK_NULL_HANDLE;
        VkDescriptorSet dset  = VK_NULL_HANDLE;
        Buffer bSamples, bCands, bCosts;
        std::atomic<bool> busy{false};
    };
    Slot slots[NSLOT];
    std::atomic<int> nslot{3};

    std::mutex submit_mu;                // vkQueueSubmit only
    std::atomic<uint64_t> n_cands{0};
    std::atomic<uint64_t> n_macs{0};   // (bsize-ord)*ord, comparable to the CPU's
    // Per-call elapsed cannot be summed once dispatches overlap -- with six
    // slots in flight that counts the same wall time up to six times. Track
    // the span from the first dispatch's start to the last one's end instead,
    // which is the window the device was actually working in.
    std::atomic<uint64_t> t_first{UINT64_MAX};
    std::atomic<uint64_t> t_last{0};
    std::chrono::steady_clock::time_point t_origin = std::chrono::steady_clock::now();
    std::atomic<size_t>   min_batch{0};
    std::atomic<int>      pcap{8};
    // Share throttle. Work is claimed greedily -- a subframe goes to the GPU
    // whenever a slot is free -- which over-commits a device slower than the
    // host: the CPU finishes its share early and idles at the DP's barrier
    // while the GPU is still working. Accepting only `duty` percent of offers
    // hands the surplus back. Deterministic (a counter, not a coin) so a run
    // stays reproducible; the output is invariant to the split either way.
    std::atomic<int>      duty{100};
    std::atomic<uint64_t> offers{0};

    static constexpr uint32_t WG = 128;  // 4 candidates per work group

    bool init();
    void destroy();
    bool grow(Buffer& b, VkDeviceSize need);
    uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags want) const;
    void bindDescriptors(Slot& s);
};

uint32_t GpuEvaluator::Impl::findMem(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

bool GpuEvaluator::Impl::grow(Buffer& b, VkDeviceSize need) {
    if (b.size >= need) return true;
    if (b.map) { vkUnmapMemory(dev, b.mem); b.map = nullptr; }
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
    b.buf = VK_NULL_HANDLE; b.mem = VK_NULL_HANDLE; b.size = 0;

    // Round up so a slowly growing workload does not reallocate every call.
    VkDeviceSize sz = 1 << 16;
    while (sz < need) sz <<= 1;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = sz;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
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
    b.size = sz;
    return true;
}

void GpuEvaluator::Impl::bindDescriptors(Slot& s) {
    VkDescriptorBufferInfo dbi[3]{};
    dbi[0].buffer = s.bSamples.buf; dbi[0].range = VK_WHOLE_SIZE;
    dbi[1].buffer = s.bCands.buf;   dbi[1].range = VK_WHOLE_SIZE;
    dbi[2].buffer = s.bCosts.buf;   dbi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wr[3]{};
    for (int i = 0; i < 3; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = s.dset;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);
}

bool GpuEvaluator::Impl::init() {
    // ---- instance --------------------------------------------------------
    // Portability enumeration exists only on layered implementations
    // (MoltenVK); enabling it unconditionally fails a native loader.
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
    std::vector<VkExtensionProperties> ie(nie);
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, ie.data());
    std::vector<const char*> iexts;
    VkInstanceCreateFlags iflags = 0;
    for (const auto& e : ie) {
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_enumeration")) {
            iexts.push_back("VK_KHR_portability_enumeration");
            iflags |= 0x00000001u;  // ENUMERATE_PORTABILITY_BIT_KHR
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

    // Prefer a discrete part; the search is compute-bound, so a dedicated GPU
    // wins whenever one exists.
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

    VkPhysicalDeviceSubgroupProperties sgp{};
    sgp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sgp;
    vkGetPhysicalDeviceProperties2(phys, &p2);

    // The kernel assigns one bit-plane per lane, so it needs exactly 32.
    if (sgp.subgroupSize != 32) {
        why = std::string(props.deviceName) + ": subgroup size is " +
              std::to_string(sgp.subgroupSize) + ", the kernel requires 32";
        return false;
    }
    const VkSubgroupFeatureFlags needsg =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    if ((sgp.supportedOperations & needsg) != needsg) {
        why = std::string(props.deviceName) +
              ": missing required subgroup operations (ballot/arithmetic/shuffle)";
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
    // A compute-only family avoids sharing the graphics queue's scheduling.
    for (uint32_t i = 0; i < nq; ++i)
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qfam = i; found = true; break; }
    if (!found)
        for (uint32_t i = 0; i < nq; ++i)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; found = true; break; }
    if (!found) { why = "no compute queue"; return false; }

    uint32_t nde = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, nullptr);
    std::vector<VkExtensionProperties> de(nde);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, de.data());
    std::vector<const char*> dexts;
    for (const auto& e : de)
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset"))
            dexts.push_back("VK_KHR_portability_subset");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures want{};
    want.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dexts.size();
    dci.ppEnabledExtensionNames = dexts.data();
    dci.pEnabledFeatures = &want;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        why = std::string(props.deviceName) + ": device creation failed"; return false;
    }
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // ---- pipeline --------------------------------------------------------
    VkDescriptorSetLayoutBinding bind[3]{};
    for (int i = 0; i < 3; ++i) {
        bind[i].binding = (uint32_t)i;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = bind;
    if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl) != VK_SUCCESS) {
        why = "descriptor set layout"; return false;
    }
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * NSLOT};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = NSLOT;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &pool) != VK_SUCCESS) {
        why = "descriptor pool"; return false;
    }
    for (int i = 0; i < NSLOT; ++i) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(dev, &dsai, &slots[i].dset) != VK_SUCCESS) {
            why = "descriptor set"; return false;
        }
    }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &plo) != VK_SUCCESS) {
        why = "pipeline layout"; return false;
    }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kSweepSpv);
    smci.pCode    = kSweepSpv;
    if (vkCreateShaderModule(dev, &smci, nullptr, &shader) != VK_SUCCESS) {
        why = "shader module"; return false;
    }
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shader;
    cpi.stage.pName = "main";
    cpi.layout = plo;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) != VK_SUCCESS) {
        why = std::string(props.deviceName) + ": compute pipeline creation failed";
        return false;
    }

    for (int i = 0; i < NSLOT; ++i) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qfam;
        if (vkCreateCommandPool(dev, &cpci, nullptr, &slots[i].pool) != VK_SUCCESS) {
            why = "command pool"; return false;
        }
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = slots[i].pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &cbai, &slots[i].cmd) != VK_SUCCESS) {
            why = "command buffer"; return false;
        }
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev, &fci, nullptr, &slots[i].fence) != VK_SUCCESS) {
            why = "fence"; return false;
        }
    }

    why = std::string(props.deviceName) + " (subgroup 32, shaderInt64)";
    ok = true;
    return true;
}

void GpuEvaluator::Impl::destroy() {
    if (dev == VK_NULL_HANDLE) { if (inst) vkDestroyInstance(inst, nullptr); return; }
    vkDeviceWaitIdle(dev);
    auto killbuf = [&](Buffer& b) {
        if (b.map) vkUnmapMemory(dev, b.mem);
        if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
        b = Buffer{};
    };
    for (int i = 0; i < NSLOT; ++i) {
        killbuf(slots[i].bSamples); killbuf(slots[i].bCands); killbuf(slots[i].bCosts);
        if (slots[i].fence) vkDestroyFence(dev, slots[i].fence, nullptr);
        if (slots[i].pool)  vkDestroyCommandPool(dev, slots[i].pool, nullptr);
    }
    if (pipe)   vkDestroyPipeline(dev, pipe, nullptr);
    if (shader) vkDestroyShaderModule(dev, shader, nullptr);
    if (plo)    vkDestroyPipelineLayout(dev, plo, nullptr);
    if (pool)   vkDestroyDescriptorPool(dev, pool, nullptr);
    if (dsl)    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (inst)   vkDestroyInstance(inst, nullptr);
}

// ---------------------------------------------------------------- public

GpuEvaluator::GpuEvaluator() : m_impl(new Impl) {
    if (!m_impl->init()) m_impl->ok = false;
}

GpuEvaluator::~GpuEvaluator() { m_impl->destroy(); }

bool GpuEvaluator::available() const { return m_impl->ok; }
void GpuEvaluator::set_min_batch(size_t n) {
    m_impl->min_batch.store(n, std::memory_order_relaxed);
}
size_t GpuEvaluator::min_batch() const {
    return m_impl->min_batch.load(std::memory_order_relaxed);
}
void GpuEvaluator::set_slots(int n) {
    m_impl->nslot.store(n < 1 ? 1 : (n > Impl::NSLOT ? Impl::NSLOT : n),
                        std::memory_order_relaxed);
}
int GpuEvaluator::slots() const {
    return m_impl->nslot.load(std::memory_order_relaxed);
}
void GpuEvaluator::set_duty(int pct) {
    m_impl->duty.store(pct < 1 ? 1 : (pct > 100 ? 100 : pct),
                       std::memory_order_relaxed);
}
int GpuEvaluator::duty() const {
    return m_impl->duty.load(std::memory_order_relaxed);
}
uint64_t GpuEvaluator::macs() const {
    return m_impl->n_macs.load(std::memory_order_relaxed);
}
bool GpuEvaluator::would_accept() const {
    Impl& I = *m_impl;
    if (!I.ok) return false;
    const int nsl = I.nslot.load(std::memory_order_relaxed);
    for (int i = 0; i < nsl; ++i)
        if (!I.slots[i].busy.load(std::memory_order_relaxed)) return true;
    return false;
}
void GpuEvaluator::set_partition_cap(int p) {
    m_impl->pcap.store(p < 1 ? 1 : (p > 8 ? 8 : p), std::memory_order_relaxed);
}
int GpuEvaluator::partition_cap() const {
    return m_impl->pcap.load(std::memory_order_relaxed);
}
const std::string& GpuEvaluator::why() const { return m_impl->why; }

void GpuEvaluator::stats(uint64_t* candidates, double* seconds) const {
    if (candidates) *candidates = m_impl->n_cands.load(std::memory_order_relaxed);
    if (seconds) {
        const uint64_t a = m_impl->t_first.load(std::memory_order_relaxed);
        const uint64_t b = m_impl->t_last.load(std::memory_order_relaxed);
        *seconds = (a == UINT64_MAX || b <= a) ? 0.0 : (double)(b - a) * 1e-6;
    }
}

bool GpuEvaluator::evaluate(const int32_t* shifted, uint32_t bsize,
                            const std::vector<Candidate>& cands,
                            std::vector<uint32_t>& out_costs) {
    Impl& I = *m_impl;
    if (!I.ok || cands.empty()) return false;
    // The kernel walks the block in fixed 32-sample chunks.
    if (bsize % 32u != 0u) return false;
    if (cands.size() < I.min_batch.load(std::memory_order_relaxed)) return false;

    const int dty = I.duty.load(std::memory_order_relaxed);
    if (dty < 100) {
        const uint64_t n = I.offers.fetch_add(1, std::memory_order_relaxed);
        if ((int)((n * 100) % 10000 / 100) >= dty) return false;
    }

    // Claim a slot. Failing is not an error: the caller encodes on the CPU
    // instead, and both paths produce the same winner, so the output is
    // unchanged either way. Idling a worker to wait for the device is strictly
    // worse than having it do the work itself.
    int si = -1;
    const int nsl = I.nslot.load(std::memory_order_relaxed);
    for (int i = 0; i < nsl; ++i) {
        bool expect = false;
        if (I.slots[i].busy.compare_exchange_strong(expect, true,
                                                    std::memory_order_acquire)) {
            si = i; break;
        }
    }
    if (si < 0) return false;
    Impl::Slot& sl = I.slots[si];
    struct Release {
        Impl::Slot& s;
        ~Release() { s.busy.store(false, std::memory_order_release); }
    } release{sl};

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t us0 = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        t0 - I.t_origin).count();
    uint64_t prev = I.t_first.load(std::memory_order_relaxed);
    while (us0 < prev &&
           !I.t_first.compare_exchange_weak(prev, us0, std::memory_order_relaxed)) {}

    const VkDeviceSize needS = (VkDeviceSize)bsize * sizeof(int32_t);
    const VkDeviceSize needC = (VkDeviceSize)cands.size() * 34 * sizeof(int32_t);
    const VkDeviceSize needO = (VkDeviceSize)cands.size() * sizeof(uint32_t);
    const bool grew = sl.bSamples.size < needS || sl.bCands.size < needC ||
                      sl.bCosts.size   < needO;
    if (!I.grow(sl.bSamples, needS)) return false;
    if (!I.grow(sl.bCands,   needC)) return false;
    if (!I.grow(sl.bCosts,   needO)) return false;
    if (grew) I.bindDescriptors(sl);

    std::memcpy(sl.bSamples.map, shifted, (size_t)needS);
    int32_t* cp = (int32_t*)sl.bCands.map;
    for (size_t i = 0; i < cands.size(); ++i) {
        cp[i * 34 + 0] = cands[i].order;
        cp[i * 34 + 1] = cands[i].shift;
        std::memcpy(&cp[i * 34 + 2], cands[i].qc, 32 * sizeof(int32_t));
    }

    PushConsts pcv{ (int32_t)cands.size(), (int32_t)bsize, 0,
                    (int32_t)I.pcap.load(std::memory_order_relaxed) };
    const uint32_t cpw    = Impl::WG / 32;               // candidates per group
    const uint32_t groups = ((uint32_t)cands.size() + cpw - 1) / cpw;

    if (vkResetCommandBuffer(sl.cmd, 0) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(sl.cmd, &bi) != VK_SUCCESS) return false;
    vkCmdBindPipeline(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, I.pipe);
    vkCmdBindDescriptorSets(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, I.plo, 0, 1,
                            &sl.dset, 0, nullptr);
    vkCmdPushConstants(sl.cmd, I.plo, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pcv, &pcv);
    vkCmdDispatch(sl.cmd, groups, 1, 1);
    if (vkEndCommandBuffer(sl.cmd) != VK_SUCCESS) return false;

    // Only the submit is serialised -- a VkQueue is externally synchronised,
    // but waiting is per-fence and must stay outside the lock or the device is
    // back to one dispatch at a time.
    {
        std::lock_guard<std::mutex> lk(I.submit_mu);
        vkResetFences(I.dev, 1, &sl.fence);
        VkSubmitInfo si2{};
        si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si2.commandBufferCount = 1;
        si2.pCommandBuffers = &sl.cmd;
        if (vkQueueSubmit(I.queue, 1, &si2, sl.fence) != VK_SUCCESS) return false;
    }
    if (vkWaitForFences(I.dev, 1, &sl.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    out_costs.resize(cands.size());
    std::memcpy(out_costs.data(), sl.bCosts.map, (size_t)needO);

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us1 = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - I.t_origin).count();
    uint64_t last = I.t_last.load(std::memory_order_relaxed);
    while (us1 > last &&
           !I.t_last.compare_exchange_weak(last, us1, std::memory_order_relaxed)) {}
    uint64_t macs = 0;
    for (const auto& cd : cands)
        macs += (uint64_t)(bsize - (uint32_t)cd.order) * (uint64_t)cd.order;
    I.n_macs.fetch_add(macs, std::memory_order_relaxed);
    I.n_cands.fetch_add(cands.size(), std::memory_order_relaxed);
    return true;
}

} // namespace flacoutcpp

#endif // FLACOUT_HAVE_VULKAN
