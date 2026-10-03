# Visual learning companion: follow the data, state, and time

This supplements the engineering roadmap. Visual explanation is a recurring part of the course, including animations where movement or changing state improves understanding. The first four interactive scenes are included in the chat and in [the standalone walkthrough](visualizations/inference-data-flow.html). After cloning, open that HTML file in your browser; GitHub displays its source rather than running it. The remaining scenes below are planned milestone deliverables, not claims that all visualizations are already implemented.

## Four walkthroughs available now

Use the topic selector, then Play, Pause, Previous, or Next. Playback stops at the final step. Under reduced-motion preferences, spatial transitions are suppressed; manual stepping remains available.

### 1. Inside a generation step

Follow input IDs through embeddings, Q/K/V projections, causal attention, the residual and MLP paths, output scores, token selection, and streaming. Shapes match the tiny engine's width of 64, four heads, and vocabulary size of 256. `T` is the number of positions supplied to this forward pass: the prompt during prefill, usually one newly selected token during cached decode.

The walkthrough expands the operations in one decoder block; Q/K/V, attention, and MLP repeat for each layer. Read this alongside `TinyDecoder.forward()` and then `generate()`. It is a simplified data-flow view, not a profile of execution time. In cached attention, the number of available key/value positions can exceed the number of new query positions.

**Predict before stepping:** Which tensor contains one score for every possible next token? Which position's scores does generation use?

### 2. Who can attend to whom?

Read a four-token causal attention matrix. Rows are queries and columns are keys; each key has an associated value vector. Later positions are masked. The highlighted row contains illustrative normalized weights for combining allowed value vectors. They are not measured model activations and do not explain a real model's reasoning.

Rows are revealed sequentially to teach the mask. Prefill can compute the prompt's rows in parallel; the animation is not claiming the model must process them one at a time. The view shows one attention head in one layer.

**Predict before stepping:** Can query t1 use value t3? Why must cached chunk attention compare absolute positions rather than start its mask again from zero?

### 3. Prefill, decode, and KV reuse

The prompt consists of abstract tokens t0, t1, and t2. Prefill stores their keys and values and selects output token g1. Decode then consumes g1, adds its keys and values, and selects g2. The output token selected at the end of a step is not yet cached; its K/V are computed when it is consumed by a later forward pass.

The upper row shows positions forwarded when the entire prefix is recomputed. The lower row distinguishes newly computed positions from earlier cached K/V that attention still reads. Read this beside the cache branch in `forward()` and the `incoming` selection in `generate()`.

**Predict before stepping:** After selecting g3, which generated tokens are already cached? Does reuse mean attention can ignore earlier keys and values? (No.)

### 4. Requests sharing a worker

A and B start in two active slots. C arrives at iteration 1 and waits. A finishes at iteration 2; C enters at iteration 3 while B continues decoding. P means prefill, D means decode, and END means the final decode step. Each prefill produces a first output token in this illustrative trace.

Columns are abstract scheduler iterations, not equal wall-clock durations. Equal-width boxes cannot establish latency or throughput gains. This is a conceptual continuous-batching example, not a benchmark or a feature already present in the starter engine. Real scheduling also needs token and memory budgets, and may split prefills into chunks.

**Predict before stepping:** When can C enter? Which request owns the released cache? What additional memory check should admission perform?

## Visual deliverables for every milestone

| Roadmap milestone | Data-flow view | Animation or interaction | Measured visual after implementation |
|---|---|---|---|
| 1. Baseline | Request → tokens → model → output | Follow a token and inspect shapes | First-token and subsequent-token timing |
| 2. Pretrained parity | Checkpoint tensor → corresponding layer | Step through layer outputs side by side | Maximum logit/layer error by layer |
| 3. Generation | Logits → probabilities → token → next input | Adjust temperature on fixed illustrative logits | Sample counts versus chosen probabilities |
| 4. KV cache | New Q/K/V plus old K/V → attention | Watch cache positions append after prefill | Used/reserved memory and decode latency versus length |
| 5. Profiling | CPU/backend work and memory movement | Highlight the operation selected in a trace | Actual operator timeline and cost breakdown |
| 6. Static batching | Separate requests → padded batch + masks | Reveal padding and finished slots | Latency and throughput across batch sizes |
| 7. Scheduling | Arrival queue → active slots → completion | Replay an identical trace under different policies | Queue wait and request-latency distributions |
| 8. Cache blocks | Logical positions → block table → physical pool | Allocate, append, free, and reuse blocks | Occupancy, unused capacity, allocation failures |
| 9. Prefix sharing | Shared prefix → diverging continuations | Reference counts, copy-on-write, and eviction | Hit rate plus first-token latency for hits/misses |
| 10. Quantization | Float weights → integer values + scales | Compare reconstructed values and error | Quality, footprint, and speed together |
| 11. Serving | Client → queue → worker → streaming response | Cancel a request and follow cleanup | Request waterfall, queue depth, and error counts |
| 12. Operations | Replicas, routing, readiness, and load | Replay bursts and worker failures | Offered load versus SLO-qualified goodput |

For optional GPU work, add memory hierarchy diagrams, kernel timelines, tiled attention animations, and transfer traces. For distributed work, add tensor partition maps and communication timelines. Label simulations separately until measured hardware traces are available.

## A consistent visual vocabulary

- **Data:** token positions, vectors, matrices, or blocks with explicit labels. Use the same token label as it moves through a scene.
- **State:** current cache length, allocated capacity, ownership, request status, and random-generator state where relevant.
- **Time:** equal spacing only for conceptual steps; measured durations must use a time axis with units.
- **Movement:** arrows show direction and moving markers show the current operation. Motion does not imply a measured duration.
- **Reuse:** distinguish existing state from newly computed state with both text and visual treatment.
- **Boundaries:** show which work belongs to the client, server, CPU, or accelerator when that distinction matters.

Keep equations close to the technical view and translate their meaning in the main narrative. For example, a heatmap should explain what its rows, columns, and colors represent before it shows numeric detail.

## Connect animations to the actual engine

The current animations are deliberately illustrative. In the KV-cache lab, add an opt-in trace callback that emits structured events such as:

```json
{"step": 1, "phase": "decode", "input_positions": [3], "cache_before": 3, "cache_after": 4, "selected_output_position": 4}
```

This example is a schema illustration. Position 3 is the first generated token being consumed, and position 4 is the next selected output. Capture events at the actual state changes instead of inferring them afterward from animation frames.

Useful fields include request ID, phase, layer, tensor name and shape, start/end position, cache block IDs, and event timestamps. For a tiny model, save selected numerical tensors when needed. Avoid recording every full activation for larger models; trace volume and tracing overhead can distort the experiment.

Save a deterministic trace and replay it visually. Check that its final state matches the engine's state. Turn detailed tracing off for performance benchmarks, or explicitly measure and report its overhead. Do not turn raw prompt logging on just to make diagrams.

## A learning session with visuals

1. Inspect the static starting state and name every object.
2. Predict what changes on the next step.
3. Advance one frame and explain any mismatch with your prediction.
4. Find the code that caused the transition.
5. Modify a small input, chunk size, or request arrival and repeat.
6. Record a correctness invariant, then measure only after it holds.
7. Capture a few representative frames and explain them in plain language for your blog.

## Blog presentation

For each post, use one overview, one focused animation, and one evidence chart when measurements exist. Additional figures should answer additional questions rather than repeat the same point. A general reader should be able to follow the story using just the overview and selected frames; an engineer should be able to inspect shapes, definitions, and raw data.

Provide descriptive captions, text alternatives, and still frames. Use labels as well as color. Keep animations user-controlled and finite. Clearly mark illustrative attention weights, invented token sequences, simulations, and actual measurements.
