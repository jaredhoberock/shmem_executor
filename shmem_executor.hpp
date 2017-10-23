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
#include "interprocess_future.hpp"
#include "socket.hpp"

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
        remote_ptr<shared_parameter_type> remote_shared_parameter(&shared_parameter<shared_parameter_type>::value.get(), 0);

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

  private:
    // a pair_factory wraps two other factories
    // and returns a pair containings their results
    template<class Factory1, class Factory2>
    struct pair_factory
    {
      mutable Factory1 factory1;
      mutable Factory2 factory2;

      std::pair<typename std::result_of<Factory1()>::type,typename std::result_of<Factory2()>::type>
        operator()() const
      {
        return std::make_pair(factory1(), factory2());
      }

      template<class OutputArchive>
      friend void serialize(OutputArchive& ar, const pair_factory& self)
      {
        ar(self.factory1, self.factory2);
      }

      template<class InputArchive>
      friend void deserialize(InputArchive& ar, pair_factory& self)
      {
        ar(self.factory1, self.factory2);
      }
    };

    // twoway_bulk_execute_functor is the functor used in twoway_bulk_execute
    // which adapts bulk_execute's one-way behavior to implement twoway_bulk_execute's
    // twoway behavior
    template<class Result, class Shared, class Function>
    struct twoway_bulk_execute_functor
    {
      mutable Function user_function;
      std::string hostname;
      int port;

      void operator()(size_t rank, remote_reference<std::pair<Result,Shared>> result_and_shared) const
      {
        // our functor receives a single shared parameter as a std::pair
        // get a raw pointer to the pair which is local to processing element 0
        std::pair<Result,Shared>* raw_ptr_to_pair = (&result_and_shared).get();

        // get raw pointers to the std::pair's two member variables
        Result* raw_ptr_to_result = &(raw_ptr_to_pair->first);
        Shared* raw_ptr_to_shared = &(raw_ptr_to_pair->second);

        // get remote_ptrs pointing to the result and shared parameter on processing element 0
        remote_ptr<Result> remote_result(raw_ptr_to_result, 0);
        remote_ptr<Shared> remote_shared_parameter(raw_ptr_to_shared, 0);

        // call the user function with the result & shared paramteter passed as remote_references
        // XXX we should catch any exception thrown from this function
        user_function(rank, *remote_result, *remote_shared_parameter);

        // wait until all processing elements have executed the user_function
        shmem_barrier_all();

        // rank 0 fulfills the promise
        if(rank == 0)
        {
          write_socket writer(hostname.c_str(), port);

          file_descriptor_ostream os(writer.get());

          interprocess_promise<Result> promise(os);

          promise.set_value(*remote_result);
        }
      }

      template<class OutputArchive>
      friend void serialize(OutputArchive& ar, const twoway_bulk_execute_functor& self)
      {
        ar(self.user_function, self.hostname, self.port);
      }

      template<class InputArchive>
      friend void deserialize(InputArchive& ar, twoway_bulk_execute_functor& self)
      {
        ar(self.user_function, self.hostname, self.port);
      }
    };

  public:
    template<class Function, class ResultFactory, class SharedFactory>
    interprocess_future<typename std::result_of<ResultFactory()>::type>
    twoway_bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      // get the name of this machine
      char hostname[HOST_NAME_MAX];
      if(gethostname(hostname, sizeof(hostname)) == -1)
      {
        throw std::system_error(errno, std::system_category(), "shmem_executor::twoway_bulk_execute(): Error after gethostname()");
      }

      int port = 71342;

      using result_type = typename std::result_of<ResultFactory()>::type;
      using shared_parameter_type = typename std::result_of<SharedFactory()>::type;

      // execute start the client process using the one-way function
      this->bulk_execute(twoway_bulk_execute_functor<result_type,shared_parameter_type,Function>{f, hostname, port}, n, pair_factory<ResultFactory,SharedFactory>{result_factory, shared_factory});

      // create a future corresponding to the client
      return interprocess_future<result_type>{read_socket(port).release()};
    }
};

// define the static member variable declared above
template<class Function, class SharedFactory>
template<class T>
uninitialized<T> shmem_executor::bulk_oneway_functor<Function,SharedFactory>::shared_parameter<T>::value;

