# Writing about LLM inference for everyone

## Your editorial promise

“I am building a small AI inference engine on my laptop to understand what happens behind a generated answer. Each post explains one idea, shows one experiment, and states what I still do not know.”

This can serve both general readers and engineers if each article has two depths: a complete main story without code, followed by an optional technical section with diagrams, formulas, code links, and reproducibility details. Do not make readers understand a tensor equation before they understand why it matters.

Use your actual engineering experience and experiments. The draft below does not invent past employment on inference teams, production scale, or results you have not measured.

## Visual storytelling throughout the series

Pair each post with a visual from the [visual learning companion](visual-learning-companion.md). Start with an overview, animate one changing mechanism, and finish with a measured chart when data exists. Keep general-reader labels visible and put tensor details in a second view or caption.

Animations must have pause and step controls, stop at the end, respect reduced-motion preferences, and remain understandable as still frames. Export a few annotated frames for readers whose blog client cannot run animation. Label invented token sequences and attention weights as illustrative; use real timestamps for any speed comparison. A moving dot represents data or a scheduling decision, never evidence that a model understands or thinks in a human way.

## A reusable article structure

1. **A familiar observation, 80–120 words.** A chatbot pauses, streams, slows down, or gives a different answer. Why should a reader care?
2. **One concrete explanation, 200–300 words.** Introduce a single mechanism with a diagram and a carefully bounded analogy.
3. **The experiment, 150–250 words.** State the hypothesis, change, measurements, and actual result. Explain every chart axis and unit.
4. **The tradeoff, 100–150 words.** What did the improvement cost? Memory, quality, fairness, or complexity?
5. **Optional technical notes.** Shapes, equations, test details, code commit, model revision, machine, and raw data.
6. **The next question, one sentence.** Link to the following experiment rather than ending with an unsupported grand claim.

These word counts are flexible. A conceptual first article does not need to fabricate an experiment result to fill the template.

## Series plan

| Post | Reader's question | Evidence to prepare | Technical appendix |
|---|---|---|---|
| 1. What happens after you press Send? | Why does an answer arrive gradually? | Request-path diagram | Forward pass and generation loop |
| 2. Where does an AI answer come from? | What are the model's numbers doing? | One token's path through your runner | Checkpoint mapping and logits parity |
| 3. Why can the same question get different answers? | Is variation a bug? | Fixed-prompt decoding comparison | Temperature, top-k/top-p, seeds |
| 4. The AI's scratchpad | How does reusing work help? | Cached/uncached correctness plus timings | KV shapes and offset masks |
| 5. Why powerful computers still wait | Why isn't raw computing power enough? | A measured profile | Memory traffic and arithmetic intensity |
| 6. One fast answer or many answers? | Why is handling a crowd harder? | Latency-throughput plot | Static and continuous batching |
| 7. How the queue changes the experience | Why do some users wait much longer? | Mixed-request timeline | Admission, fairness, deadlines |
| 8. Making room for more conversations | Why does memory fill up? | Memory pool diagram and occupancy data | Blocks, fragmentation, reference counts |
| 9. When AI can reuse earlier work | Why can repeated prompts be cheaper? | Cache hit/miss experiment | Exact prefix identity and isolation |
| 10. Smaller numbers, smaller models | What do we lose by compressing a model? | Size, quality, and speed comparison | Quantization scales and error |
| 11. More than a model | What happens when a connection breaks? | Cancellation/failure walkthrough | Streaming, cleanup, observability |
| 12. What my tiny engine taught me | What makes a system reliable? | Reproducible capstone results | SLOs, goodput, design tradeoffs |

Publish at the pace your evidence supports. A fortnightly cadence gives complex experiments more room than a rigid weekly promise.

## Translate accurately

| Technical term | Plain-language introduction | Limit of the explanation |
|---|---|---|
| Token | A chunk of text the model works with | It may be a word, part of a word, punctuation, or something else |
| Inference | Running a trained model to produce an output | The surrounding service can also retrieve documents or call tools |
| Weights | Numeric settings learned during training | They are not a simple searchable list of stored facts |
| Logits | Scores for possible next chunks | They are not calibrated truth or confidence scores |
| Prefill | Processing the input before the first generated token | It also produces the scores used to select that first token |
| Decode | Repeatedly producing the next token | This is distinct from converting token IDs back into text |
| KV cache | Saved intermediate work for the current context | It is not long-term memory or an update to the weights |
| Batch | Several requests processed together | They still need separate positions, masks, and state |
| Quantization | Representing numbers with fewer bits | Smaller storage does not automatically mean faster computation |
| Throughput | How much work finishes per second | It does not describe each user's waiting time |
| Tail latency | How long the slowest fraction of requests wait | Averages can conceal it |
| SLO | A measurable service promise | A proposed target is not a guarantee that every request meets it |

## First post draft

### What happens after you press Send? A journey inside an AI reply

You ask a chatbot a question. There is a short pause. Then the answer starts appearing, a few pieces at a time.

It feels like one action: ask a question, get an answer. Behind that experience is a sequence of smaller jobs. Some turn your words into numbers. Some run the model. Others decide when your request gets computer time and how the answer reaches your screen.

I want to understand those jobs by building a small inference engine on my laptop. An inference engine is the software that runs a trained model and manages the process of generating an output. I am starting with a tiny version whose calculations I can inspect, then working toward a system that can handle several requests.

The first useful distinction is between learning and using what was learned. During training, a model's numerical settings are adjusted using data. During ordinary inference, those settings stay fixed while the system processes a new input and produces output.

Think of a recipe whose instructions are already written. Following the recipe is different from developing it. The comparison is imperfect—an AI model contains numerical operations rather than cooking instructions—but it helps separate these two activities.

### Your words become pieces

The model does not receive a sentence exactly as you see it. A tokenizer breaks text into pieces and assigns numbers to them. These pieces are called tokens.

A token might be a word, part of a word, or punctuation. Different tokenizers divide text differently. That is why counting words is not the same as counting the amount of input a model processes.

The model turns those token numbers into numerical representations and passes them through its layers. For a common kind of text-generating model, the result includes scores for what token could come next.

### An answer grows through repeated predictions

Suppose the input begins, “On a rainy afternoon…” The model produces scores for possible next tokens. The generation software selects a token according to its settings, adds it to the sequence, and repeats the process.

Selecting the highest-scoring token is one option. Sampling among possible tokens is another. These choices influence the result, which is one reason identical questions can produce different wording.

This description concerns common autoregressive text models. A full AI application may also search documents, call software tools, or perform other work around the model. Those activities affect the experience too.

### The first piece and the next pieces involve different work

Before producing the first generated token, the model processes the input prompt. Engineers commonly call this stage **prefill**.

Once generation is underway, the system repeatedly performs **decode** steps to continue the answer. The first visible delay and the rhythm of later output therefore deserve separate measurements.

A short initial pause does not guarantee a smooth response afterward. And a service that processes many answers per second may still make some individual users wait a long time.

### Reusing work helps, but takes memory

If the system recomputed everything about the earlier text for every new token, it would repeat a great deal of work.

A KV cache stores certain intermediate numerical results so they can be reused. A rough analogy is keeping useful calculations on a scratchpad instead of starting each calculation from a blank page.

The scratchpad analogy has limits. This cache does not mean the AI has permanently learned your conversation. It stores a particular kind of intermediate state, and that state takes up memory. Longer contexts and more simultaneous requests can require more space. The [Hugging Face cache guide](https://huggingface.co/docs/transformers/main/en/cache_explanation) explains the mechanism in technical detail.

### A busy service also needs to decide who goes next

Your request may be one of many arriving at the same time. A scheduler decides which work the model should process next. It must consider available memory, waiting requests, and requests already generating answers.

Serving several requests together can make better use of the machine, but the choices involve tradeoffs. Waiting to form a group may delay someone. Giving a very long input too much uninterrupted time may stall another answer.

This is where inference becomes a systems problem: deciding how to share limited computation and memory while providing a useful experience.

### What I am building first

My first lab uses a very small decoder with untrained weights. It performs the calculations, but it is not expected to produce meaningful language.

That makes it useful for a narrower question: can I change how the computation is organized while preserving its numerical result? I can compare a version that recomputes the whole sequence with one that keeps a cache, and test whether they agree.

Useful language comes later, when I load a pretrained model and compare my implementation with a trusted reference. Performance claims will come from measured experiments with clearly stated hardware and inputs.

The next question is how much repeated work a cache removes—and how much memory we spend to avoid it.

## Technical note for the draft

The draft deliberately contains no speedup claim. The included mechanics benchmark is not evidence about a large trained model or an internet service. Before adding results to a later article, record the code version, machine, model, precision, prompt/output lengths, warmup, repeats, timing boundary, and raw data.

## Experiment record to copy into every later post

```text
Reader's question:
Hypothesis:
Code commit and dependency versions:
Model/tokenizer revision:
Hardware, RAM, backend, thread count:
Precision and cache configuration:
Prompt lengths, output lengths, workload/arrival distribution:
What changed:
What stayed fixed:
Correctness checks:
Warmup and number of measurements:
Timing boundary and units:
Raw data location:
Observed result:
Tradeoff / alternative explanation:
What this experiment does not establish:
Next question:
```

## Before publishing

Check that every “I measured” statement has data and every “I built” statement describes code that exists. Separate calculations, simulations, and hardware measurements. Label illustrative numbers. Explain unfamiliar terms on first use. Name the scope of analogies. Do not turn a model's next-token score into a claim of factual confidence, describe caching as permanent learning, or imply that a laptop benchmark establishes datacenter performance.

For a general reader, the main story should stand alone if they skip every code block. For an engineer, the optional notes should provide enough detail to reproduce or challenge the result.
