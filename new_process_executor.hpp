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

#include <limits.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>

extern char** environ;

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <mutex>
#include <cassert>
#include <system_error>


#include "interprocess_future.hpp"
#include "active_message.hpp"


namespace this_process
{


static pid_t get_id()
{
  return getpid();
}


static const std::vector<std::string>& environment()
{
  static std::vector<std::string> result;

  if(result.empty())
  {
    for(char** variable = environ; *variable; ++variable)
    {
      result.push_back(std::string(*variable));
    }
  }

  return result;
}


static const std::string& filename()
{
  static std::string result;

  if(result.empty())
  {
    std::string symbolic_name = std::string("/proc/") + std::to_string(getpid()) + "/exe";

    char real_name[PATH_MAX + 1];
    ssize_t length = readlink(symbolic_name.c_str(), real_name, PATH_MAX);
    if(length == -1)
    {
      throw std::runtime_error("this_process::filename(): Error after readlink().");
    }

    real_name[length] = '\0';

    result = real_name;
  }

  return result;
}


}


// this tracks all processes created through process_executors
// and blocks on their completion in its destructor
class process_context
{
  public:
    inline ~process_context()
    {
      wait();
    }

    template<class Function>
    void execute(const char* launcher_program_filename, const char** launcher_program_argv, Function&& f)
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // create an active_message out of f
      active_message message(decay_copy(std::forward<Function>(f)));

      // make a copy of this process's environment and set the variable ALTERNATE_MAIN_ACTIVE_MESSAGE to contain serialized message
      auto spawnee_environment = this_process::environment();
      set_variable(spawnee_environment, "EXECUTE_ACTIVE_MESSAGE_BEFORE_MAIN", to_string(message));
      auto spawnee_environment_view = environment_view(spawnee_environment);


      // concatenate launchee program filename onto launcher_argv
      std::vector<const char*> args;

      // to assemble the arguments vector, start with the launcher program's arguments
      for(const char** arg = launcher_program_argv; *arg != nullptr; ++arg)
      {
        args.push_back(*arg);
      }

      // follow with the name of the launchee program and end with nullptr
      args.push_back(this_process::filename().c_str());
      args.push_back(nullptr);

      // spawn the process
      pid_t spawnee_id;
      int error = posix_spawnp(&spawnee_id, launcher_program_filename, nullptr, nullptr, const_cast<char**>(args.data()), spawnee_environment_view.data());
      if(error)
      {
        throw std::system_error(errno, std::generic_category(), "process_context::execute(): Error after posix_spawn()");
      }

      // keep track of the new process
      processes_.push_back(spawnee_id);
    }

    template<class Function>
    interprocess_future<
      typename std::result_of<typename std::decay<Function>::type()>::type
    >
      twoway_execute(const char* launcher_program_filename, const char** launcher_program_argv, Function&& f)
    {
      // create a pipe
      int in_and_out_file_descriptors[2];
      if(pipe(in_and_out_file_descriptors) == -1)
      {
        throw std::runtime_error("process_context::twoway_execute(): Error after pipe()");
      }

      int in = in_and_out_file_descriptors[0];
      int out = in_and_out_file_descriptors[1];

      // wrap f in a function which writes its result to the output pipe
      invoke_and_write_result<typename std::decay<Function>::type> g{std::forward<Function>(f), out};

      // ensure that the input file is closed in the spawned process
      int flags = fcntl(in, F_GETFD);
      if(flags == -1)
      {
        throw std::runtime_error("process::twoway_executor(): Error after fctnl()");
      }

      flags |= FD_CLOEXEC;
      if(fcntl(in, F_SETFD, flags) == -1)
      {
        throw std::runtime_error("process::twoway_executor(): Error after fctnl()");
      }

      // execute the wrapped function
      execute(launcher_program_filename, launcher_program_argv, std::move(g));

      // close the output descriptor in this process
      close(out);

      // return a future
      return interprocess_future<int>(in);
    }

    inline void wait()
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // wait for each spawned process to finish
      for(pid_t p : processes_)
      {
        waitpid(p, nullptr, 0);
      }

      processes_.clear();
    }

  private:
    template<class Function>
    struct invoke_and_write_result
    {
      mutable Function f;
      int file_descriptor;

      void operator()() const
      {
        // invoke f
        auto result = f();

        // create an interprocess_promise corresponding to our file_descriptor
        file_descriptor_ostream os(file_descriptor);
        assert(os.good());
        interprocess_promise<decltype(result)> promise(os);

        // set the promise's value
        promise.set_value(std::move(result));

        // close the file
        // XXX interprocess_promise should close the file in set_value probably
        ::close(file_descriptor);
      }

      template<class OutputArchive>
      friend void serialize(OutputArchive& ar, const invoke_and_write_result& self)
      {
        ar(self.f, self.file_descriptor);
      }

      template<class InputArchive>
      friend void deserialize(InputArchive& ar, invoke_and_write_result& self)
      {
        ar(self.f, self.file_descriptor);
      }
    };

    static inline void set_variable(std::vector<std::string>& environment, const std::string& variable, const std::string& value)
    {
      auto existing_variable = std::find_if(environment.begin(), environment.end(), [&](const std::string& current_variable)
      {
        // check if variable is a prefix of current_variable
        auto result = std::mismatch(variable.begin(), variable.end(), current_variable.begin());
        if(result.first == variable.end())
        {
          // check if the next character after the prefix is an equal sign
          return result.second != variable.end() && *result.second == '=';
        }
    
        return false;
      });
    
      if(existing_variable != environment.end())
      {
        *existing_variable = variable + "=" + value;
      }
      else
      {
        environment.emplace_back(variable + "=" + value);
      }
    }
    
    static inline std::vector<char*> environment_view(const std::vector<std::string>& environment)
    {
      std::vector<char*> result;
    
      for(const std::string& variable : environment)
      {
        result.push_back(const_cast<char*>(variable.c_str()));
      }

      // the view is assumed to be null-terminated
      result.push_back(0);
    
      return result;
    }

    template<class Arg>
    static typename std::decay<Arg>::type decay_copy(Arg&& arg)
    {
      return std::forward<Arg>(arg);
    }

    std::mutex mutex_;
    std::vector<pid_t> processes_;
};

process_context global_process_context;


// this replaces a process's execution of main() with an active_message if
// the environment variable EXECUTE_ACTIVE_MESSAGE_BEFORE_MAIN is defined
struct execute_active_message_before_main_if
{
  execute_active_message_before_main_if()
  {
    char* variable = std::getenv("EXECUTE_ACTIVE_MESSAGE_BEFORE_MAIN");
    if(variable)
    {
      active_message message = from_string<active_message>(variable);
      message.activate();

      std::exit(EXIT_SUCCESS);
    }
  }
};

execute_active_message_before_main_if before_main{};


class new_process_executor
{
  public:
    template<class RangeOfConstChar>
    explicit new_process_executor(const char* launcher_program_filename, const RangeOfConstChar& launcher_program_argv)
      : launcher_program_filename_(launcher_program_filename),
        launcher_program_argv_(launcher_program_argv.begin(), launcher_program_argv.end())
    {}

    new_process_executor()
      : new_process_executor("/usr/bin/env", std::array<const char*,2>{"/usr/bin/env", nullptr})
    {}

    template<class Function>
    void execute(Function&& f) const
    {
      global_process_context.execute(launcher_program_filename_.c_str(), const_cast<const char**>(launcher_program_argv_.data()), std::forward<Function>(f));
    }
    
    template<class Function>
    interprocess_future<
      typename std::result_of<typename std::decay<Function>::type()>::type
    >
      twoway_execute(Function&& f) const
    {
      return global_process_context.twoway_execute(launcher_program_filename_.c_str(), const_cast<const char**>(launcher_program_argv_.data()), std::forward<Function>(f));
    }

  private:
    std::string launcher_program_filename_;
    std::vector<const char*> launcher_program_argv_;
};

