# shmem_executor
An executor which creates execution on OpenSHMEM processing elements

# Demo

`shmem_executor` can create groups of concurrent execution agents executing on remote nodes:

```
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

  if(idx == 0)
  {
    result = 7;
  }
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

  std::cout << "OK" << std::endl;
}

```

Example program output:

```
# compile demo program with oshc++
$ oshc++ -std=c++11 demo.cpp -o demo
# start an interactive slurm session, ask to reserve 4 nodes
$ srun --pty --qos=big -n 4 /bin/bash -i
# run the demo program
$ ./demo
```

