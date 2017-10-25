# shmem_executor
An executor which creates execution on OpenSHMEM processing elements

# Demo

`shmem_executor` can create groups of concurrent execution agents executing on remote nodes:

```c++
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
