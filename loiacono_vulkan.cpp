#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t SIGNAL_LENGTH = 8192;
constexpr uint32_t INPUT_SAMPLES = SIGNAL_LENGTH;
constexpr uint32_t FREQUENCY_BANDS = 512;
constexpr uint32_t THREADS_PER_WORKGROUP = 64;
constexpr uint32_t WORKGROUP_COUNT_X =
    (FREQUENCY_BANDS + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP;

struct BufferResource {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
};

inline void checkVk(VkResult result, const std::string &msg) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(msg + " (vk result: " + std::to_string(result) + ")");
  }
}

std::vector<char> readFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open shader file: " + path);
  }
  auto size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type.");
}

BufferResource createBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties) {
  BufferResource resource{};
  resource.size = size;
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  checkVk(vkCreateBuffer(device, &bufferInfo, nullptr, &resource.buffer),
          "create buffer");

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, resource.buffer, &memRequirements);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
                                            properties);
  checkVk(vkAllocateMemory(device, &allocInfo, nullptr, &resource.memory),
          "allocate buffer memory");
  vkBindBufferMemory(device, resource.buffer, resource.memory, 0);
  return resource;
}

void copyToBuffer(VkDevice device, const BufferResource &resource, const void *data,
                  VkDeviceSize size) {
  void *mapped = nullptr;
  vkMapMemory(device, resource.memory, 0, size, 0, &mapped);
  std::memcpy(mapped, data, static_cast<size_t>(size));
  vkUnmapMemory(device, resource.memory);
}

int main() {
  try {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "LoiaconoGPU";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vulkanese";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    checkVk(vkCreateInstance(&instanceInfo, nullptr, &instance), "create instance");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      throw std::runtime_error("No Vulkan physical devices available.");
    }
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    VkPhysicalDevice physicalDevice = physicalDevices[0];

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             queueFamilies.data());

    int computeFamilyIndex = -1;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeFamilyIndex = static_cast<int>(i);
        break;
      }
    }
    if (computeFamilyIndex < 0) {
      throw std::runtime_error("Could not find compute queue family.");
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device;
    checkVk(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device), "create logical device");

    VkQueue computeQueue;
    vkGetDeviceQueue(device, static_cast<uint32_t>(computeFamilyIndex), 0, &computeQueue);

    VkCommandPool commandPool;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    checkVk(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool),
            "create command pool");

    auto shaderCode = readFile("shaders/loiacono.comp.spv");
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModInfo{};
    shaderModInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModInfo.codeSize = shaderCode.size();
    shaderModInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
    checkVk(vkCreateShaderModule(device, &shaderModInfo, nullptr, &shaderModule),
            "create shader module");

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.reserve(3);
    for (uint32_t binding = 0; binding < 3; ++binding) {
      VkDescriptorSetLayoutBinding layoutBinding{};
      layoutBinding.binding = binding;
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layoutBinding.descriptorCount = 1;
      layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      layoutBinding.pImmutableSamplers = nullptr;
      layoutBindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    checkVk(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout),
            "create descriptor set layout");

    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    checkVk(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
            "create pipeline layout");

    VkPipeline pipeline;
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    checkVk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
            "create compute pipeline");

    VkDescriptorPool descriptorPool;
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = 1;
    descriptorPoolInfo.pPoolSizes = &poolSize;
    descriptorPoolInfo.maxSets = 1;
    checkVk(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool),
            "create descriptor pool");

    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    checkVk(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet),
            "allocate descriptor set");

    BufferResource xBuffer = createBuffer(device, physicalDevice,
                                          static_cast<VkDeviceSize>(INPUT_SAMPLES * sizeof(float)),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    BufferResource lBuffer = createBuffer(device, physicalDevice,
                                          static_cast<VkDeviceSize>(FREQUENCY_BANDS * sizeof(float)),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    BufferResource fBuffer = createBuffer(device, physicalDevice,
                                          static_cast<VkDeviceSize>(FREQUENCY_BANDS * sizeof(float)),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::vector<float> signal(INPUT_SAMPLES);
    constexpr float sampleRate = 48000.0f;
    constexpr float amplitude = 0.5f;
    for (uint32_t i = 0; i < INPUT_SAMPLES; ++i) {
      float t = static_cast<float>(i) / sampleRate;
      signal[i] = amplitude * (std::sin(2.0f * 3.1415926f * 440.0f * t) +
                               0.35f * std::sin(2.0f * 3.1415926f * 880.0f * t));
    }
    copyToBuffer(device, xBuffer, signal.data(), xBuffer.size);

    std::vector<float> fprimes(FREQUENCY_BANDS);
    const float minFreq = 100.0f / sampleRate;
    const float maxFreq = 12000.0f / sampleRate;
    for (uint32_t i = 0; i < FREQUENCY_BANDS; ++i) {
      float ratio = static_cast<float>(i) / static_cast<float>(FREQUENCY_BANDS - 1);
      float logMin = std::log(minFreq);
      float logMax = std::log(maxFreq);
      fprimes[i] = std::exp(logMin + (logMax - logMin) * ratio);
    }
    copyToBuffer(device, fBuffer, fprimes.data(), fBuffer.size);

    std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
    bufferInfos[0].buffer = xBuffer.buffer;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = xBuffer.size;
    bufferInfos[1].buffer = fBuffer.buffer;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = fBuffer.size;
    bufferInfos[2].buffer = lBuffer.buffer;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = lBuffer.size;

    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
    for (uint32_t i = 0; i < descriptorWrites.size(); ++i) {
      descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[i].dstSet = descriptorSet;
      descriptorWrites[i].dstBinding = i;
      descriptorWrites[i].dstArrayElement = 0;
      descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[i].descriptorCount = 1;
      descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    checkVk(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer),
            "allocate command buffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                            &descriptorSet, 0, nullptr);
    vkCmdDispatch(commandBuffer, WORKGROUP_COUNT_X, 1, 1);
    vkEndCommandBuffer(commandBuffer);

    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    checkVk(vkCreateFence(device, &fenceInfo, nullptr, &fence), "create fence");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    checkVk(vkQueueSubmit(computeQueue, 1, &submitInfo, fence), "submit compute work");
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);

    void *mappedL = nullptr;
    vkMapMemory(device, lBuffer.memory, 0, lBuffer.size, 0, &mappedL);
    auto *result = reinterpret_cast<float *>(mappedL);
    std::cout << "Spectrum (all bins):\n";
    for (uint32_t i = 0; i < FREQUENCY_BANDS; ++i) {
      std::cout << "bin " << i << ": " << result[i] << "\n";
    }
    vkUnmapMemory(device, lBuffer.memory);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    auto destroyBuffer = [&](const BufferResource &res) {
      vkDestroyBuffer(device, res.buffer, nullptr);
      vkFreeMemory(device, res.memory, nullptr);
    };
    destroyBuffer(xBuffer);
    destroyBuffer(fBuffer);
    destroyBuffer(lBuffer);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
