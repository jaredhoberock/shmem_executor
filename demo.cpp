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

// To compile:
//
//     $ clang -I../../openshmem-am-root/include -std=c++11 demo.cpp -lstdc++ -L../../openshmem-am-root/lib -lopenshmem -lelf -L../../gasnet-root/lib -lgasnet-smp-par -lpthread -lrt
// or
//
//     $ ../../openshmem-am-root/bin/oshc++ -std=c++11 demo.cpp

#include <iostream>
#include <cassert>

#include "shmem_executor.hpp"

void hello(int idx, remote_reference<int> shared_parameter)
{
  std::cout << "hello world from processing element " << idx << ", received " << shared_parameter << std::endl;
  assert(shared_parameter == 13);
}

void twoway_hello(int idx, remote_reference<int> result, remote_reference<int> shared_parameter)
{
  std::cout << "hello world from processing element " << idx << ", received " << shared_parameter << std::endl;
  assert(shared_parameter == 13);

  if(idx == 0)
  {
    result = 7;
  }
}

void exceptional_hello(int idx, remote_reference<int> result, remote_reference<int> shared_parameter)
{
  std::cout << "hello world from processing element " << idx << ", received " << shared_parameter << std::endl;
  assert(shared_parameter == 13);

  throw std::runtime_error("exception");
}

int factory()
{
  return 13;
}

int main()
{
  shmem_executor exec;

  // test one-way execution
  exec.bulk_execute(hello, 2, factory);

  // test two-way execution
  interprocess_future<int> result = exec.twoway_bulk_execute(twoway_hello, 2, factory, factory);
  assert(result.get() == 7);

  // test two-way exceptional execution
  interprocess_future<int> exceptional_result = exec.twoway_bulk_execute(exceptional_hello, 2, factory, factory);

  try
  {
    exceptional_result.get();

    // ensure we didn't produce a result
    assert(0);
  }
  catch(interprocess_exception e)
  {
    std::cerr << "Caught exception: [" << e.what() << "]" << std::endl;
  }
  catch(...)
  {
    // ensure we didn't catch some other type of exception
    assert(0);
  }

  std::cout << "OK" << std::endl;
}

