# heinsen_sequence

Sample code for computing the sequence $x_t = a_t x_{t-1} + b_t$ efficiently in parallel, given $t = (1, 2, \dots, n)$, $a_t \in \mathbb{R}^n$, $b_t \in \mathbb{R}^n$, and initial value $x_0 \in \mathbb{R}$. See ["Efficient Parallelization of a Ubiquitious Sequential Computation" (Heinsen, 2023)](http://arxiv.org/abs/2311.06281).

Sequences of this form are ubiquitous in science and engineering. For example, in the natural sciences, these sequences model quantities or populations that decay or grow by a varying rate $a_t > 0$ between net inflows or outflows $b_t$ at each time step $t$. In economics and finance, these sequences model investments that earn a different rate of return $a_t = (1 + r_t)$ between net deposits or withdrawals $b_t$ at the beginning of each time period $t$. In engineering applications, these sequences are often low-level components of larger models, *e.g.*, linearized recurrent neural networks (LRNNs) whose layers decay token features in a sequence of tokens.

It's common to see code that computes sequences of this form one element at a time. Read on to find out how to compute them efficiently in parallel!

All snippets of code assume you have a working installation of Python 3.8+ with PyTorch 2.1+.


## First, Try the Sequential Approach

The following snippet of code computes 10,000,000 elements, one element at a time. Copy and paste it to execute it. Warning: It will be painfully slow. Execution time is linear in the number of elements, $\mathcal{O}(n)$. If you get tired of waiting, interrupt execution:

```python
import torch

def naively_compute_sequentially(coeffs, values):
    x = [values[0]]  # x_0
    for a, b in zip(coeffs, values[1:]):
        x.append(a * x[-1] + b)
    return torch.stack(x)

device = 'cuda:0'     # change as necessary
seq_len = 10_000_000  # change as you wish

# Generate some random input data:
coeffs = torch.randn(seq_len, device=device)
values = torch.randn(1 + seq_len, device=device)  # includes initial value

# Compute the sequence:
x = naively_compute_sequentially(coeffs, values)  # includes initial value
```

Note: Even if you rewrite the above snippet of interpreted Python code as efficient GPU code (say, with [Triton](https://triton-lang.org)), execution will still be slow, because all elements are computed sequentially, which is inefficient in a GPU.


## Now Try Our Proposed Parallel Approach

The snippets of code below execute the same computations *in parallel* -- or more precisely, as a composition of two [prefix sums](https://en.wikipedia.org/wiki/Prefix_sum), each of which is executable in parallel. (See also [this post on implementing parallel prefix sum in CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).) The first snippet is for the general case in which $a_t \in \mathbb{R}^n$, $b_t \in \mathbb{R}^n$, and initial value $x_0 \in \mathbb{R}$. The second snippet is for the special case in which none of the inputs are negative. Copy and paste each snippet of code to run it. The difference in execution time compared to sequential computation will be quickly evident. Given $n$ parallel processors, execution time is logarithmic in the number of elements, $\mathcal{O}(\log n)$. For details on how parallelization works, including its mathematical proof, please see our preprint.


### General Case: If Any Input Can Be Negative

If any inputs are negative, we must first cast them to complex numbers before computing their logarithms:

```python
import torch
import torch.nn.functional as F

def compute_in_parallel(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-1), (1, 0))              # eq (2) in paper
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=-1)  # eq (7) in paper
    log_x = a_star + log_x0_plus_b_star                                   # eq (1) in paper
    return torch.exp(log_x).real

device = 'cuda:0'     # change as necessary
seq_len = 10_000_000  # change as you wish

# Generate some random input data:
coeffs = torch.randn(seq_len, device=device); 
coeffs = coeffs.masked_fill(coeffs == 0, 1e-5)    # eps for numerical stability
values = torch.randn(1 + seq_len, device=device)  # negative or positive values

# Compute the sequence:
x = compute_in_parallel(
    log_coeffs=coeffs.to(dtype=torch.complex64).log(),  # logs of coeffs < 0 are complex
    log_values=values.to(dtype=torch.complex64).log(),  # logs of values < 0 are complex
)                                                       # output includes initial value
```


### Special Case: If No Inputs Are Negative

If no inputs are negative, we don't need to cast them to complex numbers before computing their logarithms:

```python
import torch
import torch.nn.functional as F

def compute_in_parallel_special_case(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-1), (1, 0))              # eq (2) in paper
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=-1)  # eq (7) in paper
    log_x = a_star + log_x0_plus_b_star                                   # eq (1) in paper
    return torch.exp(log_x)                                               # already a float

device = 'cuda:0'     # change as necessary
seq_len = 10_000_000  # change as you wish

# Generate some random input data:
coeffs = torch.rand(seq_len, device=device) + 1e-5   # eps for numerical stability
values = torch.rand(1 + seq_len, device=device) * 3  # all values >= 0

# Compute the sequence:
x = compute_in_parallel_special_case(
    log_coeffs=coeffs.log(),  # no need to cast to complex
    log_values=values.log(),  # no need to cast to complex
)                             # output includes initial value
```


## Caching State for Incremental Computation

For computing the sequence incrementally, chunk by chunk, we recommend you cache each chunk's final log-element, `log_x[-1]`, and use it as the subsequent chunk's initial log-value, `log_values[0]`. Caching state in the domain of logarithms is more numerically stable.


## Considerations for Using in Production

The snippets of code above are meant to be *easy-to-follow recipes*. For use in production, make sure to compute all logarithms with the most efficient and numericaly stable methods available. For example, if the coefficients are gating probabilities computed from given logits, you should use `F.logsigmoid(logits)` instead of `torch.log(F.sigmoid(logits))` to compute the log-coefficients. If one of the input sequences has no negative numbers, don't cast it to complex in advance; instead, wait to cast its logarithms until after they have been summed. If you are using lower precision, don't assume numerical stability; instead, make sure both input sequences will always be within acceptable bounds. Use your common sense.

In certain production environments, it may be more efficient to represent each complex number as a (float, int) tuple to take advantage of the fact that all sums of imaginary components in our proposed method are multiples of $i \pi$, given that the logarithm of a negative number $-|x|$ is equal to the logarithm of its absolute value plus $i \pi$, _i.e._, $\log(-|x|) = \log(|x|) + i \pi$ (see [here](https://math.stackexchange.com/questions/2089690/log-of-a-negative-number)). Moreover, for any multiple $n$ of $i \pi$, we have $\exp(n i \pi) = -1$ if $n$ is odd, $1$ otherwise, so you could make the int element $n$ a boolean value in {0, 1}, and use XOR instead of $+$ as the binary associative operator in all sums of imaginary components. Keep in mind that any possible gains in efficiency will be offset by greater implementation complexity.

As always, you should test all available options to find out which one will work best for your use case.


## Compared to Blelloch's Classic Work

See [this thread](https://github.com/glassroom/heinsen_sequence/issues/1).


## Citing

```
@misc{heinsen2023parallelization,
      title={Efficient Parallelization of a Ubiquitous Sequential Computation}, 
      author={Franz A. Heinsen},
      year={2023},
      eprint={2311.06281},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```


## Notes

We originally conceived and implemented these methods as part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code, clean it up, and release it as stand-alone open-source software without having to disclose any key intellectual property.

We hope others find our work and our code useful.
