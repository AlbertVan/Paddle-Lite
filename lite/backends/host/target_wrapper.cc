// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/core/target_wrapper.h"
#include <cstring>
#include <memory>
// #include <unistd.h>
// #include <stdlib.h>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <map>

// #define PAGE_SIZE (4096)

// #ifndef ROUND_UP
// #define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
// #endif
namespace paddle {
namespace lite {

const int MALLOC_ALIGN = 64;
const int MALLOC_EXTRA = 64;

// static std::map<void*, size_t> mmap_list;
void* TargetWrapper<TARGET(kHost)>::Malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  CHECK(size);
  CHECK_GT(offset + size, size);
  size_t extra_size = sizeof(int8_t) * MALLOC_EXTRA;
  auto sum_size = offset + size;
  CHECK_GT(sum_size + extra_size, sum_size);

  // void* p = nullptr;
  // if (sum_size > 11059100) {
  //   sum_size = ROUND_UP(sum_size, PAGE_SIZE);
  //   std::cout << "mmap size = " << sum_size << std::endl;
  //   p = mmap(NULL, sum_size,  PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  //   if (p == MAP_FAILED) {
  //     fprintf(stderr, "CPU memory not enough.\n");
  //     exit(1);
  //   }
  //   std::cout << "malloc use mmap" << std::endl;
  //   /* use huge page */
  //   int ret = madvise(p, sum_size, MADV_HUGEPAGE);
  //   if (ret) {
  //       fprintf(stderr, "madvise()= %d, no enough transparent huge page available, "
  //               "please check |cat /sys/kernel/mm/transparent_hugepage/enabled|, "
  //               "measured dma speed may be slower than expected !!!\n");
  //   }
  //   mmap_list.insert(std::pair<void*, size_t>(p, sum_size));
  // } else {


  char* p = static_cast<char*>(malloc(sum_size + extra_size));
  CHECK(p) << "Error occurred in TargetWrapper::Malloc period: no enough for "
              "mallocing "
           << size << " bytes.";
  // }
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}
void TargetWrapper<TARGET(kHost)>::Free(void* ptr) {
  if (ptr) {
  //   if (mmap_list.find(static_cast<void**>(ptr)[-1]) != mmap_list.end()) {
  //     size_t size = mmap_list[static_cast<void**>(ptr)[-1]];
  //     munmap(static_cast<void**>(ptr)[-1], size);
  //     std::cout << "free mmap here" << std::endl;
  //   } else {
    free(static_cast<void**>(ptr)[-1]);
  //   }
  }
}
void TargetWrapper<TARGET(kHost)>::MemcpySync(void* dst,
                                              const void* src,
                                              size_t size,
                                              IoDirection dir) {
  if (size > 0) {
    CHECK(dst) << "Error: the destination of MemcpySync can not be nullptr.";
    CHECK(src) << "Error: the source of MemcpySync can not be nullptr.";
    memcpy(dst, src, size);
  }
}

}  // namespace lite
}  // namespace paddle
