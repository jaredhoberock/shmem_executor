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

#include <vector>
#include <string>

#include <limits.h>
#include <unistd.h>

extern char** environ;

namespace this_process
{


static inline pid_t get_id()
{
  return getpid();
}


static inline const std::vector<std::string>& environment()
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


static inline const std::string& filename()
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

