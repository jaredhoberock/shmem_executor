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
#include "remote_ptr.hpp"
#include "uninitialized.hpp"

class shmem_executor
{
  private:
    template<class Function, class SharedFactory>
    struct bulk_oneway_functor
    {
      mutable Function f;
      mutable SharedFactory shared_factory;

      // XXX in C++17, this would just be a variable template
      template<class T>
      struct shared_parameter
      {
        static uninitialized<T> value;
      };

      template<class T, __REQUIRES(std::is_trivially_destructible<T>::value)>
      static void synchronize_and_destroy_shared_parameter_if(int)
      {
        // no-op
      }

      template<class T, __REQUIRES(!std::is_trivially_destructible<T>::value)>
      static void synchronize_and_destroy_shared_parameter_if(int rank)
      {
        shmem_barrier_all();

        if(rank == 0)
        {
          shared_parameter<T>::value.destroy();
        }
      }

      void operator()() const
      {
        // construct OpenSHMEM
        shmem_init();

        // compute the type of the shared parameter
        using shared_parameter_type = typename std::result_of<SharedFactory()>::type;

        // get this processing element's rank
        int rank = shmem_my_pe();

        // rank 0 initializes the shared parameter as an OpenSHMEM "symmetric" object
        if(rank == 0)
        {
          // note that there is only one of these "symmetric" objects per-type, per-process
          // however, since shmem_executor spawns a process for each agent it creates,
          // this is safe
          //
          // in other words, this function, operator(), is the moral equivalent of main()
          shared_parameter<shared_parameter_type>::value.emplace(shared_factory());
        }

        // all processing elements wait for the shared_parameter to be constructed
        shmem_barrier_all();

        // point at PE 0's instance of shared_parameter
        remote_ptr<int> remote_shared_parameter(&shared_parameter<shared_parameter_type>::value.get(), 0);

        // invoke f, passing a remote_reference to the shared parameter
        f(rank, *remote_shared_parameter);

        // synchronize with a barrier and destroy the shared parameter if it has a non-trivial destructor
        synchronize_and_destroy_shared_parameter_if<shared_parameter_type>(rank);

        // destroy OpenSHMEM
        shmem_finalize();
      }

      template<class OutputArchive>
      friend void serialize(OutputArchive& ar, const bulk_oneway_functor& self)
      {
        ar(self.f);
        ar(self.shared_factory);
      }

      template<class InputArchive>
      friend void deserialize(InputArchive& ar, bulk_oneway_functor& self)
      {
        ar(self.f);
        ar(self.shared_factory);
      }
    };

  public:
    template<class Function, class SharedFactory>
    void bulk_execute(Function f, size_t n, SharedFactory shared_factory) const
    {
      std::string n_as_string = std::to_string(n);
      std::array<const char*, 4> argv = {"oshrun", "-n", n_as_string.c_str(), nullptr};
      new_process_executor exec(argv[0], argv);

      exec.execute(bulk_oneway_functor<Function, SharedFactory>{f, shared_factory});
    }
};

// define the static member variable declared above
template<class Function, class SharedFactory>
template<class T>
uninitialized<T> shmem_executor::bulk_oneway_functor<Function,SharedFactory>::shared_parameter<T>::value;

