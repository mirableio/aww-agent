## **Lightweight Agent Framework**

## Requirements Document

### **Problem Statement**

Production AI agents require an execution loop that calls LLMs, executes tools based on model decisions, and continues until a task completes. Existing solutions fall into two categories: (1) heavy frameworks (LangChain, CrewAI) that obscure control flow and complicate debugging, or (2) raw SDKs that require reimplementing the same patterns repeatedly.

We need a minimal, transparent agent harness that provides essential infrastructure (context management, provider abstraction, observability) without hiding the execution loop. Target users are developers building production agents who need full control over behavior while avoiding boilerplate.

### **Design Principles**

**Transparency over magic.** Every decision point (when to stop, what to compact, how to route) must be explicit and overridable. No implicit callbacks or middleware chains.

**Provider-agnostic core.** Canonical message format converts bidirectionally to Anthropic, OpenAI, Bedrock, Vertex. Switching providers requires only adapter swap, not loop rewrite.

**Composition over inheritance.** Each component (context manager, stop evaluator, tool registry) is independent and replaceable. Framework provides defaults; users can substitute any piece.

### **Core Components**

**1\. Message Abstraction**

Canonical format representing conversation turns. Must capture: role (system/user/assistant/tool), content (text or structured blocks), tool calls (id, name, arguments), tool results (id, content, error flag). Metadata: token count, timestamp, importance score for compaction decisions.

Adapters convert to/from provider formats. Key differences to handle: Anthropic uses content blocks with `type: tool_use` inside assistant messages and requires tool results as user messages; OpenAI uses separate `tool_calls` array and dedicated `tool` role. System prompt handling differs (separate param vs. message).

**2\. Provider Adapter**

Abstract interface with implementations per provider. Methods: `complete()` (single turn), `stream()` (token-level), `count_tokens()`, `convert_messages()`. Adapters handle authentication, retries, rate limiting. Initial targets: Anthropic (primary), OpenAI, Google.

**3\. Tool Registry**

Tools defined as: name, description, schema parameters (pydantic model), handler function. Metadata: timeout, confirmation requirement, parallelization hint (independent/sequential). Registry validates schemas, dispatches execution, captures results. Handlers are async callables returning string or structured content.

**4\. Context Manager**

Maintains conversation history. Tracks token counts per message. Triggers compaction when approaching threshold (configurable, default 70-80% of model's context window).

Compaction strategies:

* **Summarization**: LLM-generated summary of older turns, preserving recent N messages (counted by token limit threshold) verbatim. Structured format with mandatory sections (goal, decisions, artifacts, state, next steps) prevents information loss.  
* **Search**: Dedicated tool to search this thread history, in order to simplify we avoid embeddings in this phase and use keyword search (LLM decides a set of keywords for OR search).

Compaction can use a different model than main agent (cheaper/faster). Summary history preserved for debugging.

**5\. Stop Evaluator**

Determines when loop terminates. Conditions (checked in order):

* **Natural completion**: Model returns response without tool calls — primary signal.  
* **Explicit done tool**: Model calls designated completion tool (e.g., `submit_result(answer)`). Use when structured output required.  
* **Max iterations**: Safety ceiling (default 25). Prevents runaway loops.  
* **Token budget**: Cumulative spend limit for cost control.  
* **Error threshold**: N consecutive tool failures → abort.  
* **User interrupt**: Human-in-the-loop signal.

Returns reason for stop, enabling different handling per case.

**6\. Agent Loop**

Orchestrates the cycle: receive user input → get context → call model → check stop → execute tools → loop. Yields messages as produced for streaming UX. Manages state transitions and coordinates all components.

### **Streaming**

Two levels required:

**Turn-level streaming**: Yield complete messages as they're produced. Sufficient for most backend uses. Simple to implement via async generator from loop.

**Token-level streaming**: Yield tokens/chunks as model generates. Required for real-time UX (typing indicator, progressive display). Complications: tool calls arrive as partial JSON that must be buffered until complete; some providers send tool calls only after text content finishes.

Implementation: Adapter's `stream()` method yields chunks. Loop buffers tool call fragments, emits text chunks immediately, yields complete tool calls when fully received. Consumer can process text progressively while tools execute only on completion.

### **Parallel Tool Execution**

Not all tools must run sequentially. When model requests multiple tools in single turn:

**Classification**: Tools marked as `independent` (read-only, no side effects) can parallelize. Tools marked `sequential` (writes, state mutations) run in order.

**Execution**: Group independent tools → `asyncio.gather()`. Sequential tools run in order after parallel batch completes. Mixed: parallelize independent subset, then sequential.

**Result ordering**: Preserve original tool call order in results regardless of execution order — models expect consistent ordering.

**Timeout handling**: Per-tool timeouts. If one parallel tool hangs, others complete; failed tool returns error result. Don't block entire batch.

### **Observability**

Debugging agent runs requires structured logging at multiple levels:

**Event types**: Loop start/end, model call (request/response), tool execution (start/end/error), compaction triggered, stop condition hit, checkpoint saved.

**Per-event data**:

* Model calls: messages sent, tokens in/out, latency, model used, cache hit  
* Tool execution: tool name, arguments, result (truncated), duration, error details  
* Compaction: trigger reason, messages compacted, summary generated, token reduction  
* Stop: reason, iteration count, final state

**Integration points**: Pluggable sink interface. Implementations for: structured logging (JSON lines), OpenTelemetry spans, custom callbacks. Default: console logger with configurable verbosity.

**Trace correlation**: Run ID propagated through all events. Parent-child relationships for nested calls (e.g., sub-agent invocations).

**Sensitive data**: Tool arguments/results may contain PII. Configurable redaction patterns. Default: truncate large outputs, redact known patterns (API keys, emails).

### **State Checkpoints**

Long-running agents (especially with expensive models like Opus) need crash recovery:

**What to checkpoint**:

* Full message history (pre-compaction)  
* Compaction summaries generated  
* Current iteration count, error counts  
* Tool execution state (for idempotency)  
* Cost accumulator

**When to checkpoint**:

* After each complete loop iteration (model call \+ tool execution)  
* Before compaction (preserve full context)  
* On graceful shutdown signal

**Storage interface**: Abstract persistence layer. Implementations: local filesystem (JSON), Redis, S3, database. Keyed by run ID \+ sequence number.

**Recovery**: On startup, check for incomplete run. Load latest checkpoint, resume from last stable state. Handle partial tool execution (either re-run with idempotency or skip if side-effecting).

**Retention**: Configurable. Keep last N checkpoints per run. Purge on successful completion (optional).

### **Cost Tracking**

Token-based billing requires accurate accounting:

**Metrics tracked**:

* Input tokens per model call  
* Output tokens per model call  
* Cached tokens (Anthropic prompt caching)  
* Total tokens per run  
* Estimated cost (requires price table per model)

**Granularity**: Per-call, per-iteration, per-run totals. Breakdown by: main model vs. compaction model, user messages vs. tool results.

**Budget enforcement**: Optional hard limit. Check before each model call. If projected call would exceed budget, stop loop with `TOKEN_BUDGET` reason.

**Price tables**: Configurable mapping of model ID → (input\_price\_per\_1k, output\_price\_per\_1k). Updated separately from code. Defaults for common models.

**Reporting**: Expose via observability events. Final run summary includes total cost. Integration with billing systems via webhook/callback.

### **Non-Goals (Explicit Exclusions)**

* Multi-agent orchestration (composition of multiple loops is user's responsibility)  
* Built-in tool implementations (users provide handlers)  
* Prompt management/templating (separate concern)  
* Vector stores / RAG (use external libraries)  
* UI components  
* Automatic retry strategies beyond basic provider-level retries

### **Success Criteria**

1. Developer can implement a working agent in \<50 lines using framework  
2. Switching from Anthropic to OpenAI requires changing one line (adapter instantiation)  
3. Full run is reproducible from checkpoint after crash  
4. Cost of 1000-turn agent run is trackable to ±5% accuracy  
5. Any loop iteration can be debugged from logs alone without reproduction

### **Suggested Implementation Order**

1. Message abstraction \+ Anthropic adapter (minimal working loop)  
2. Tool registry \+ execution  
3. Stop evaluator (all conditions)  
4. Observability (logging foundation)  
5. Context manager \+ compaction  
6. Token-level streaming  
7. Parallel tool execution  
8. Additional provider adapters  
9. Cost tracking  
10. State checkpoints

