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

#include <shmem.h>

#include "new_process_executor.hpp"

class shmem_executor
{
  private:
    template<class Function>
    struct initialize_and_invoke_with_index_and_then_finalize
    {
      mutable Function f;

      void operator()() const
      {
        shmem_init();

        f(shmem_my_pe());

        shmem_finalize();
      }

      template<class OutputArchive>
      friend void serialize(OutputArchive& ar, const initialize_and_invoke_with_index_and_then_finalize& self)
      {
        ar(self.f);
      }

      template<class InputArchive>
      friend void deserialize(InputArchive& ar, initialize_and_invoke_with_index_and_then_finalize& self)
      {
        ar(self.f);
      }
    };

  public:
    template<class Function>
    void execute(Function f, size_t n) const
    {
      std::string n_as_string = std::to_string(n);
      std::array<const char*, 4> argv = {"/home/jhoberock/dev/openshmem-am-root/bin/oshrun", "-n", n_as_string.c_str(), nullptr};
      new_process_executor exec(argv[0], argv);

      exec.execute(initialize_and_invoke_with_index_and_then_finalize<Function>{f});
    }
};

