# Propriocept: Physical Awareness Layer for Local LLM Inference

## 1. Purpose

This document describes a modular system for giving a local language model a form of **physical self-awareness** by observing, tokenising, predicting, and comparing the physical telemetry generated during inference.

The system treats hardware behavior as a temporal signal:

```text
prompt → inference → machine state changes → telemetry stream → event tokens
```

These event tokens can then be used to train a predictor model that learns:

```text
Given this prompt, what physical events will this machine likely experience while answering?
```

The system is not designed to produce raw monitoring dashboards only. Its goal is to translate physical machine behavior into a **semantically rich event language** that can be embedded, compared, predicted, and injected back into a model context.

---

# 2. High-Level Architecture

```text
┌────────────────────┐
│      Prompt        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│    LLM Inference   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     Telemetry      │
│ internal/external  │
└─────────┬──────────┘
          │ raw streams
          ▼
┌────────────────────┐
│     Tokeniser      │
│ event generation   │
└─────────┬──────────┘
          │ event tokens
          ▼
┌────────────────────┐
│ Event Sequence Log │
└─────────┬──────────┘
          │
          ├──────────────► dataset for Predictor
          │
          ▼
┌────────────────────┐
│ Semantic Comparator│
└────────────────────┘
```

Prediction loop:

```text
prompt
  │
  ▼
Predictor
  │ predicted event sequence
  ▼
LLM context injection / avatar state / comparison target
  │
  ▼
actual inference
  │
  ▼
actual event sequence
  │
  ▼
Semantic Comparator
  │
  ▼
prediction error / surprise / strain / mood state
```

---

# 3. Core Design Principles

## 3.1 Telemetry is raw

The telemetry module should not interpret data.

It should only collect and stream raw readings from internal and external sources.

Examples:

```text
CPU temperature: 78.2°C
GPU usage: 91%
fan speed: 4300 RPM
microphone sample: float32 audio buffer
EM probe sample: float32 signal buffer
token latency: 81 ms
```

Interpretation belongs to the Tokeniser.

---

## 3.2 Tokens should be event-based, not sample-based

The Tokeniser should avoid producing a continuous stream of numeric values like:

```text
TEMP_71, TEMP_72, TEMP_73, TEMP_74
```

Instead, it should emit meaningful events when thresholds, transitions, rhythms, or anomalies are detected:

```text
The GPU temperature crossed 80°C for the first time during the response.
The fan speed began rising sharply shortly after the model started generating tokens.
A high-frequency electromagnetic burst appeared just before the response slowed down.
```

This allows the physical signal to become closer to language.

---

## 3.3 Time should be described richly

Events should not only include timestamps.

Each event should describe its temporal position relative to multiple anchors:

- absolute wall-clock time
- monotonic session time
- inference phase
- token index
- response character span
- CPU cycle estimate if available
- previous event
- next event if post-processing is allowed
- session-defined “periods”
- user-defined temporal categories

Example:

```text
At 14.82 seconds after inference started,
during the late-middle phase of the response,
after approximately 940 generated tokens,
shortly after the first fan-speed increase,
and just before the second latency spike,
the GPU temperature crossed 90°C.
```

The goal is to place the event into a very specific region of latent space.

---

## 3.4 Event descriptions should be semantically dense

An event should include:

- what happened
- where it happened
- when it happened
- what changed before and after
- how strong the change was
- whether it was expected
- what else was happening nearby

Example:

```text
During the final third of the model response, 22.4 seconds after inference began,
while token latency was already elevated and the fan was increasing speed,
the GPU temperature crossed the critical threshold of 90°C for the first time in this session.
This marked a transition from high thermal pressure to near-throttling thermal stress.
```

This is intentionally verbose.

The goal is not compression.  
The goal is semantic specificity.

---

# 4. Module 1: Telemetry

## 4.1 Responsibility

The Telemetry module collects raw internal and external data streams independently of inference logic.

It must:

- expose internal OS-readable telemetry
- expose external real-time sensor streams
- support configurable sampling rates
- stream data on demand
- timestamp all samples using a shared monotonic clock
- avoid semantic interpretation
- provide raw samples to downstream modules

---

## 4.2 Internal Telemetry

Internal telemetry includes values readable from the operating system or hardware APIs.

Possible streams:

```text
cpu_temperature
gpu_temperature
cpu_usage
gpu_usage
vram_usage
ram_usage
fan_speed
power_draw
battery_state
disk_io
network_io
process_list
per_process_cpu_usage
per_process_memory_usage
token_latency
tokens_per_second
model_context_size
model_backend
```

Implementation should provide adapters for platform-specific sources.

Examples:

```text
Linux:
- psutil
- lm-sensors
- nvidia-smi
- rocm-smi
- /sys/class/thermal
- /proc/stat
- /proc/[pid]/stat

macOS:
- powermetrics
- iStat-compatible interfaces if available
- psutil
- Metal performance counters where available

Windows:
- WMI
- OpenHardwareMonitor / LibreHardwareMonitor
- NVIDIA Management Library
- psutil
```

---

## 4.3 External Telemetry

External telemetry is an abstract real-time data stream.

Examples:

```text
microphone input
electromagnetic probe
software-defined radio stream
accelerometer
temperature sensor
camera-derived light changes
ambient room sensor
```

The system should not assume one specific external source.

External telemetry should expose a common interface:

```text
start()
stop()
read_frame()
sampling_rate
channel_count
sample_format
```

For example, an EM probe and microphone can both be treated as time-series signal sources.

---

## 4.4 Telemetry Sample Format

Each raw telemetry sample should use a normalized envelope:

```json
{
  "stream_id": "gpu_temperature",
  "source_type": "internal",
  "timestamp_monotonic_ns": 1829381293812,
  "timestamp_wall": "2026-05-04T15:41:03.221Z",
  "sampling_rate_hz": 10,
  "unit": "celsius",
  "value": 78.4,
  "metadata": {
    "device": "GPU 0",
    "backend": "nvidia-smi"
  }
}
```

For frame-based signal sources:

```json
{
  "stream_id": "em_probe",
  "source_type": "external",
  "timestamp_monotonic_ns": 1829381293812,
  "timestamp_wall": "2026-05-04T15:41:03.221Z",
  "sampling_rate_hz": 48000,
  "unit": "amplitude",
  "value": [0.012, 0.018, 0.011],
  "metadata": {
    "device": "USB audio interface",
    "channel": 1,
    "frame_size": 1024
  }
}
```

---

## 4.5 Telemetry API

Suggested interface:

```python
class TelemetryStream:
    stream_id: str
    source_type: str
    sampling_rate_hz: float

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> dict:
        pass
```

Aggregator:

```python
class TelemetryAggregator:
    def register_stream(self, stream: TelemetryStream) -> None:
        pass

    def start_all(self) -> None:
        pass

    def stop_all(self) -> None:
        pass

    def subscribe(self, stream_ids: list[str] | None = None):
        """
        Returns iterator or async stream of raw telemetry samples.
        """
        pass
```

---

# 5. Module 2: Tokeniser

## 5.1 Responsibility

The Tokeniser converts raw telemetry streams into semantically rich event descriptions.

It should:

- consume raw telemetry samples
- detect threshold crossings
- detect trend changes
- detect anomalies
- detect correlations between streams
- generate verbose natural-language events
- attach structured metadata to each event
- preserve links to raw samples
- support configurable event rules

---

## 5.2 Event-Driven Tokenisation

Instead of producing tokens at every sample, the Tokeniser emits events when meaningful changes occur.

Event types may include:

```text
threshold_crossing
trend_started
trend_stopped
rate_of_change_spike
anomaly_detected
oscillation_detected
stability_period_started
stability_period_ended
correlation_detected
phase_transition
session_milestone
external_signal_burst
token_latency_spike
background_process_interference
thermal_throttling_warning
```

---

## 5.3 Event Rule Example

Structured rule:

```json
{
  "rule_id": "gpu_temp_cross_90",
  "stream_id": "gpu_temperature",
  "type": "threshold_crossing",
  "operator": ">=",
  "threshold": 90,
  "direction": "rising",
  "cooldown_ms": 5000,
  "severity": "critical",
  "description_template": "The GPU temperature crossed {threshold}°C for the first time during {session_period}, while {concurrent_conditions}."
}
```

---

## 5.4 Temporal Context Builder

The Tokeniser should include a Temporal Context Builder that describes when an event occurred using multiple anchors.

Temporal anchors:

```text
wall_clock_time
monotonic_session_time
time_since_inference_start
time_since_prompt_received
time_since_first_token
time_until_inference_end
token_index_nearest
response_phase
session_phase
previous_event_relation
next_event_relation
period_label
cycle_estimate
```

Example output:

```json
{
  "absolute": "2026-05-04T15:41:19.812Z",
  "session_time": "00:03:22.418 since session start",
  "inference_time": "14.82 seconds after inference began",
  "token_position": "near generated token 940",
  "response_phase": "late-middle phase of the response",
  "relative_previous_event": "8.1 seconds after the first fan-speed increase",
  "relative_next_event": "2.4 seconds before the second latency spike",
  "period_label": "during the warming period of this inference session",
  "cycle_estimate": "after approximately 48.2 billion CPU reference cycles"
}
```

Note: `relative_next_event` requires post-processing after the event sequence is complete.

---

## 5.5 Session Periods

The system should support arbitrary period definitions.

Examples:

```text
cold_start
warmup
stable_generation
thermal_rise
sustained_pressure
near_throttle
cooldown
background_interference
recovery
```

Periods may be:

- rule-based
- user-defined
- inferred from telemetry
- predicted by another model

Periods are useful because they create meaningful temporal anchors beyond raw timestamps.

---

## 5.6 Event Object Format

Each generated event should have both structured and natural-language forms.

```json
{
  "event_id": "evt_000128",
  "run_id": "run_2026_05_04_154103",
  "event_type": "threshold_crossing",
  "severity": "critical",
  "stream_refs": ["gpu_temperature", "token_latency", "fan_speed"],
  "timestamp_monotonic_ns": 1829396113812,
  "temporal_context": {
    "absolute": "2026-05-04T15:41:19.812Z",
    "inference_time": "14.82 seconds after inference began",
    "token_position": "near generated token 940",
    "response_phase": "late-middle phase",
    "period_label": "thermal rise period",
    "relative_previous_event": "shortly after the first fan-speed increase"
  },
  "structured_payload": {
    "stream_id": "gpu_temperature",
    "value": 90.1,
    "threshold": 90,
    "unit": "celsius",
    "direction": "rising"
  },
  "natural_language": "During the late-middle phase of the response, 14.82 seconds after inference began and near generated token 940, the GPU temperature crossed 90°C for the first time in this run. This happened shortly after the first fan-speed increase, while token latency was already elevated, marking a transition into near-throttling thermal stress.",
  "raw_sample_refs": ["sample_018291", "sample_018292"]
}
```

---

## 5.7 Token Sequence Format

The Tokeniser should emit a sequence like:

```text
<BODY_SEQUENCE_START>
During the early response phase, 1.2 seconds after inference began, the CPU package power rose sharply from idle to active inference load.
During the early-middle response phase, near generated token 180, the fan speed began increasing for the first time in this run.
During the late-middle response phase, 14.82 seconds after inference began and near generated token 940, the GPU temperature crossed 90°C for the first time in this run.
During the final response phase, token latency spiked while the thermal state remained near throttling pressure.
<BODY_SEQUENCE_END>
```

This sequence can be used directly as text input/output for model training.

---

# 6. Module 3: Detokeniser

## 6.1 Responsibility

The Detokeniser reconstructs approximate telemetry streams from event sequences.

It should:

- parse event descriptions or structured event objects
- infer key telemetry points
- interpolate values between events
- reconstruct approximate time-series streams
- expose uncertainty
- allow visual replay of predicted physical state

The Detokeniser does not need to perfectly recreate raw telemetry.  
Its purpose is to create a plausible physical curve from discrete semantic events.

---

## 6.2 Detokenisation Strategy

Input:

```text
GPU temperature crossed 90°C at 14.82 seconds.
Fan speed began rising sharply at 8.1 seconds.
Token latency spiked during the final phase.
```

Output:

```json
{
  "streams": {
    "gpu_temperature": [
      {"t": 0.0, "value": 62.0, "confidence": 0.4},
      {"t": 8.1, "value": 78.0, "confidence": 0.5},
      {"t": 14.82, "value": 90.0, "confidence": 0.95},
      {"t": 22.0, "value": 91.5, "confidence": 0.6}
    ],
    "fan_speed": [
      {"t": 0.0, "value": 1800, "confidence": 0.4},
      {"t": 8.1, "value": 2600, "confidence": 0.8},
      {"t": 22.0, "value": 4200, "confidence": 0.5}
    ]
  }
}
```

Interpolation methods:

```text
linear
step
spline
exponential rise
thermal curve model
custom per-stream model
```

---

# 7. Module 4: Predictor

## 7.1 Responsibility

The Predictor estimates which physical events are likely to happen during inference for a given prompt and initial machine state.

It should output a semantically rich event sequence.

Input:

```text
prompt
initial telemetry snapshot
model identity
hardware identity
runtime configuration
recent session context
```

Output:

```text
predicted event sequence
```

---

## 7.2 Training Dataset

Each training sample should contain:

```json
{
  "run_id": "run_2026_05_04_154103",
  "prompt": "Explain transformer attention in detail.",
  "model_response": "...",
  "initial_telemetry_snapshot": {
    "gpu_temperature": 61.2,
    "cpu_temperature": 55.0,
    "fan_speed": 1800,
    "gpu_usage": 4,
    "ram_usage": 62
  },
  "runtime_config": {
    "model": "local-llm-name",
    "quantization": "Q4_K_M",
    "context_length": 8192,
    "backend": "llama.cpp",
    "device": "GPU"
  },
  "event_sequence": [
    {
      "event_id": "evt_000001",
      "natural_language": "During the early response phase, 1.2 seconds after inference began, the CPU package power rose sharply from idle to active inference load."
    }
  ]
}
```

---

## 7.3 Training Formats

### Prompt-only prediction

```text
Input:
Prompt + initial telemetry snapshot

Output:
Event sequence
```

This is the most interesting version because it asks the system to predict the physical future before response generation.

---

### Prompt-response prediction

```text
Input:
Prompt + generated response + initial telemetry snapshot

Output:
Event sequence
```

This is easier and useful for early training because response length and complexity are known.

---

### Streaming prediction

```text
Input:
Prompt + partial response + telemetry so far

Output:
Next likely physical event
```

This enables real-time body prediction.

---

## 7.4 Predictor Output Example

```text
<BODY_SEQUENCE_START>
During the early response phase, shortly after inference begins, the GPU usage will rise rapidly from idle into sustained active generation.
During the early-middle response phase, after the first several hundred generated tokens, the GPU temperature will enter a thermal rise period.
During the late-middle response phase, if the answer continues beyond approximately one thousand tokens, token latency may begin increasing while fan speed rises.
During the final response phase, the system will likely remain thermally stable below critical throttling pressure.
<BODY_SEQUENCE_END>
```

---

# 8. Module 5: Semantic Comparator

## 8.1 Responsibility

The Semantic Comparator compares predicted and actual event sequences using embeddings rather than raw numeric equality.

It should:

- convert event descriptions into embedding vectors
- compare predicted vs actual event sequences
- compute semantic distance
- align events temporally and semantically
- identify missing, unexpected, delayed, or exaggerated events
- output comparison metrics

---

## 8.2 Why Semantic Comparison

The system should not only compare:

```text
predicted GPU temp = 89
actual GPU temp = 90
```

It should compare meaning:

```text
Predicted: The machine entered high thermal pressure late in the response.
Actual: The GPU crossed into near-throttling stress during the late-middle response.
```

These are semantically close even if numeric values differ.

Likewise:

```text
Predicted: Stable generation.
Actual: Background process caused latency spikes.
```

These are semantically far apart.

---

## 8.3 Comparator Pipeline

```text
predicted event sequence
        │
        ▼
event segmentation
        │
        ▼
embedding generation
        │
        ▼
semantic alignment
        │
        ▼
distance calculation
        │
        ▼
comparison report
```

---

## 8.4 Comparison Metrics

Suggested metrics:

```text
semantic_similarity_mean
semantic_similarity_min
temporal_alignment_error
missing_event_count
unexpected_event_count
severity_error
phase_error
stream_coverage_score
prediction_confidence
surprise_score
strain_score
```

---

## 8.5 Event Alignment

Events should be aligned by a combination of:

- embedding similarity
- event type
- stream references
- severity
- temporal phase
- token position
- causal relation

Example:

```json
{
  "predicted_event_id": "pred_evt_004",
  "actual_event_id": "actual_evt_007",
  "semantic_similarity": 0.86,
  "temporal_alignment_error_seconds": 3.2,
  "phase_alignment": "same broad phase, actual happened earlier",
  "interpretation": "The predictor correctly anticipated thermal pressure but underestimated how soon it would appear."
}
```

---

# 9. Inference Runtime Flow

## 9.1 Data Collection Mode

Used to build training data.

```text
1. Receive prompt.
2. Capture initial telemetry snapshot.
3. Start telemetry streams.
4. Run LLM inference.
5. Record token timings and response.
6. Stop telemetry streams.
7. Tokenise raw telemetry into event sequence.
8. Store prompt, response, telemetry, and events as one run.
```

---

## 9.2 Prediction Mode

Used once the Predictor exists.

```text
1. Receive prompt.
2. Capture initial telemetry snapshot.
3. Predictor generates expected body event sequence.
4. Optionally inject predicted sequence into LLM context.
5. Start telemetry streams.
6. Run LLM inference.
7. Tokenise actual telemetry into actual event sequence.
8. Compare predicted and actual event sequences.
9. Output response plus body comparison metrics.
10. Drive avatar mood from actual telemetry and prediction error.
```

---

# 10. Storage Model

## 10.1 Run Record

```json
{
  "run_id": "run_2026_05_04_154103",
  "created_at": "2026-05-04T15:41:03.221Z",
  "prompt": "...",
  "response": "...",
  "initial_telemetry_snapshot": {},
  "runtime_config": {},
  "raw_telemetry_uri": "runs/run_2026_05_04_154103/raw_telemetry.jsonl",
  "events_uri": "runs/run_2026_05_04_154103/events.jsonl",
  "body_sequence_text_uri": "runs/run_2026_05_04_154103/body_sequence.txt",
  "comparison_uri": null
}
```

---

## 10.2 Raw Telemetry Storage

Use JSONL for early prototyping:

```jsonl
{"stream_id":"gpu_temperature","timestamp_monotonic_ns":1000,"value":61.2,"unit":"celsius"}
{"stream_id":"gpu_temperature","timestamp_monotonic_ns":2000,"value":61.4,"unit":"celsius"}
{"stream_id":"fan_speed","timestamp_monotonic_ns":2000,"value":1800,"unit":"rpm"}
```

For high-rate external streams, use binary formats:

```text
WAV / FLAC for audio-like signals
NumPy arrays
Parquet
HDF5
Zarr
```

---

# 11. Suggested Implementation Structure

```text
physical_awareness/
  telemetry/
    __init__.py
    base.py
    aggregator.py
    internal/
      cpu.py
      gpu.py
      memory.py
      fan.py
      process.py
    external/
      audio.py
      em_probe.py
      generic_stream.py

  tokeniser/
    __init__.py
    rules.py
    event.py
    temporal_context.py
    period_detector.py
    tokenizer.py

  detokeniser/
    __init__.py
    parser.py
    interpolator.py
    stream_reconstructor.py

  predictor/
    __init__.py
    dataset.py
    train.py
    infer.py
    prompts.py

  comparator/
    __init__.py
    embedder.py
    aligner.py
    metrics.py
    comparator.py

  runtime/
    __init__.py
    collect_run.py
    predict_run.py
    inference_wrapper.py
    context_injection.py

  storage/
    __init__.py
    run_store.py
    schemas.py

  avatar/
    __init__.py
    state_mapper.py
    websocket_server.py

  configs/
    telemetry.yaml
    tokeniser_rules.yaml
    runtime.yaml
```

---

# 12. MVP Scope

The first MVP should avoid EM complexity and focus on proving the loop.

## MVP Modules

Required:

```text
Telemetry:
- CPU temperature
- GPU temperature
- CPU usage
- GPU usage
- RAM usage
- token latency

Tokeniser:
- threshold crossing events
- rising/falling trend events
- latency spike events
- simple temporal context

Storage:
- run records
- raw telemetry JSONL
- event JSONL
- body sequence text

Comparator:
- embedding-based similarity between predicted and actual event text

Predictor:
- initially can be a placeholder LLM prompt
- later fine-tuned model
```

Optional in MVP:

```text
fan speed
power draw
external microphone
EM probe
detokeniser
avatar websocket
```

---

# 13. Non-Goals for MVP

Do not initially attempt:

```text
perfect hardware accuracy
real-time EM decoding
full biometric-style emotional modeling
claiming consciousness or subjective experience
complex multimodal fusion
highly optimized binary storage
```

The MVP should prove:

```text
Can we turn machine telemetry into meaningful event-language?
Can we predict that event-language from a prompt?
Can we compare predicted and actual physical events semantically?
Can this drive an expressive avatar state?
```

---

# 14. Example End-to-End Run

## Input Prompt

```text
Write a detailed explanation of transformer attention with examples.
```

## Initial Telemetry

```json
{
  "gpu_temperature": 62.4,
  "cpu_temperature": 51.2,
  "fan_speed": 1900,
  "gpu_usage": 3,
  "cpu_usage": 12,
  "ram_usage": 68
}
```

## Actual Event Sequence

```text
<BODY_SEQUENCE_START>
During the first second after inference began, the GPU usage rose sharply from idle to sustained active generation.
During the early response phase, near generated token 150, the GPU temperature began a steady upward trend from 62°C.
During the middle response phase, 9.4 seconds after inference began and near generated token 640, the fan speed began increasing for the first time in this run.
During the late-middle response phase, 16.8 seconds after inference began and near generated token 1120, token latency increased while GPU temperature remained under 85°C.
During the final response phase, the system remained thermally stable and did not enter throttling pressure.
<BODY_SEQUENCE_END>
```

## Predicted Event Sequence

```text
<BODY_SEQUENCE_START>
Shortly after inference begins, the GPU will move from idle into sustained active generation.
During the early-middle portion of the response, thermal pressure will rise gradually as the answer length increases.
During the later portion of the response, fan speed may increase while token generation remains stable.
The run will likely complete without reaching critical thermal pressure.
<BODY_SEQUENCE_END>
```

## Comparator Output

```json
{
  "semantic_similarity_mean": 0.84,
  "missing_event_count": 1,
  "unexpected_event_count": 0,
  "temporal_alignment_error_mean_seconds": 2.7,
  "surprise_score": 0.22,
  "strain_score": 0.48,
  "interpretation": "The predictor correctly anticipated active generation, gradual thermal rise, fan increase, and absence of throttling. It did not explicitly predict the late token latency increase."
}
```

---

# 15. Avatar State Mapping

The comparator and telemetry can produce a compact avatar state.

```json
{
  "calm": 0.62,
  "strain": 0.48,
  "surprise": 0.22,
  "fatigue": 0.31,
  "confidence": 0.84,
  "thermal_pressure": 0.57,
  "background_interference": 0.08
}
```

Suggested mappings:

```text
high strain → tighter posture, faster breathing, warmer color
high surprise → eye widening, glitch pulse, sudden motion
high fatigue → slower blink, drooping, reduced movement
high confidence → smoother animation, synchronized breathing
high thermal pressure → glow, sweat, red/orange tint
background interference → twitch, distraction, visual noise
```

---

# 16. Open Design Questions

These should remain configurable, not hardcoded:

```text
How verbose should event descriptions be?
Should event language be deterministic templates or LLM-generated?
How much raw numeric data should be included inside event text?
Should the Predictor see the expected response length?
Should body predictions be injected into the main LLM context?
Should event descriptions include future-relative anchors after post-processing?
How should arbitrary session periods be defined?
Which embedding model should be used for semantic comparison?
How should EM/audio features be discretized into event language?
```

---

# 17. Recommended First Build

The best first build is:

```text
Python service
JSONL storage
psutil-based internal telemetry
optional NVIDIA telemetry
llama.cpp or Ollama inference wrapper
rule-based tokeniser
template-based event language
embedding comparator
simple WebSocket avatar state output
```

This will be enough to produce the first living version of the system:

```text
a local model whose machine-body is observed,
translated into language,
predicted from prompts,
compared against reality,
and expressed as mood.
```