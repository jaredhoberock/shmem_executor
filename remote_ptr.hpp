// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <type_traits>
#include <shmem.h>

#include "pointer_adaptor.hpp"

#define __REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

class remote_memory_accessor
{
  public:
    remote_memory_accessor(int processing_element)
      : processing_element_(processing_element)
    {}

    int processing_element() const
    {
      return processing_element_;
    }

    template<class T,
             __REQUIRES(
               std::is_default_constructible<T>::value
               //and std::is_trivially_copyable<T>::value
            )>
    T load(const T* ptr) const
    {
      T result;
      shmem_getmem(&result, ptr, sizeof(T), processing_element());
      return result;
    }

    template<class T, __REQUIRES(std::is_trivially_copyable<T>::value)>
    void store(T* ptr, const T& value) const
    {
      shmem_putmem(ptr, &value, sizeof(T), processing_element());
    }

  private:
    int processing_element_;
};

template<class T>
class remote_ptr : public pointer_adaptor<T, remote_memory_accessor>
{
  private:
    using super_t = pointer_adaptor<T, remote_memory_accessor>;

  public:
    remote_ptr(T* address, int processing_element)
      : super_t(address, remote_memory_accessor(processing_element))
    {}
};

template<class T>
using remote_reference = typename remote_ptr<T>::reference;
                
