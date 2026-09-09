# Design Decisions

These are the foundational rules of the experiment. Changing them will ruin the fairness of the test by introducing hidden variables.

## Attention Dropout is Disabled

We disable attention dropout (setting it to zero) for all models. Different models handle dropout in completely different ways. If we turned it on, each model would be penalized or regularized differently, making it impossible to tell if a performance difference was due to the model's core design or just how it handled dropout. We keep dropout turned on for the parts of the neural network that all models share equally.

## Precision Settings

FlashAttention physically requires 16-bit precision to run, while the other models normally run in 32-bit. We decided to let FlashAttention run in 16-bit and treat that precision drop as an inherent part of the FlashAttention package. When we write the report, we must clarify that any speedup seen in FlashAttention is a combination of its mathematical design and its lower precision format, as we cannot separate the two.

## Paired Initialization

Every model must start with the exact same random starting weights. To guarantee this, the system generates the shared weights first in a strict order before any specific model adds its own unique parts. If a specific model like Linformer needs extra random numbers, it uses an isolated random number generator. This prevents it from stealing numbers from the main sequence and throwing the rest of the models out of sync.

## Padding and Filtering

When grouping data, we only pad sequences to match the longest item in that specific group, which saves computing power compared to padding everything to the absolute maximum limit. If a sequence is longer than our maximum limit, we delete it entirely instead of chopping off the end. Chopping off the end of a math problem changes the answer, which ruins the data. This means we throw away a lot of long sequences, which is a flaw we must admit in the final report.

## Flash Strict Mode

The system is programmed to crash immediately if FlashAttention fails to load properly. Without this strict rule, the system might quietly fall back to a slower, standard method while still labeling the results as FlashAttention. Crashing loudly ensures our speed metrics for FlashAttention are completely genuine.
