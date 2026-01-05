Phase 0 Prompt (FINAL — READY TO SHIP)
Phase 0 Task Prompt for AI Developer
(CGW Baseline + Logging + Semantic Validation)
You are an AI software engineer. Implement Phase 0 (K=1) of a Cognitive Global Workspace (CGW) system with real semantics, not placeholders.
If any requirement below is ambiguous or cannot be implemented faithfully, STOP and ask for clarification.
Do NOT invent simplified substitutes.
A. Objective
Build a single-lineage CGW baseline that:
Executes a recurrent reasoning loop over multiple steps
Uses learned routing between competing specialists
Uses functional FastMem (read/write with real data dependency)
Logs interpretable metrics per step and per run
Automatically evaluates whether the model is competent enough to proceed
This phase exists to answer one question only:
Does a single CGW instance exhibit non-trivial reasoning behaviour with meaningful routing and memory usage?
A1. CGW Model Requirements (MANDATORY)
The CGW model must exhibit these behaviours, not just expose interfaces.
1. Routing (Must Be Real and State-Dependent)
Routing must satisfy all of the following:
Routing decision is a function of the current workspace state S_t
Implemented via learned Gumbel-Softmax or softmax gating
Routing is state-dependent:
Different workspace states → may produce different routing decisions
Same workspace state (deterministically) → same routing decision
Routing must not be:
uniform random
round-robin
hardcoded to a fixed sequence
Important:
Routing is allowed to select the same specialist multiple times if the state demands it.
The requirement is state dependence, not forced variation.
Minimum acceptable implementation
routing_logits = W @ S_t
routing_probs = gumbel_softmax(routing_logits, tau)
selected_module = argmax(routing_probs)
2. Specialists (Minimum Viable Set)
You must implement four specialists, each with distinct behaviour:
Specialist	Responsibility
io	Encode input tokens / observations
logic	Apply learned transformation to workspace
mem	Read/write FastMem
sup	Emit halting signal or confidence
Each specialist must:
Read S_t
Produce an output that meaningfully changes S_{t+1}
3. FastMem (Must Be Functional)
FastMem cannot be a stub.
Requirements:
Supports write(key, value)
Supports read(key) → value
Values read must influence computation
At least one specialist must:
write to memory
later read the same key
FastMem must produce non-zero reads during successful runs.
4. Workspace Semantics
Workspace S_t must be:
persistent across steps
updated based on selected specialist output
State updates must not be:
identity functions
pure noise
single-pass feed-forward without recurrence
❗ Forbidden
If you cannot implement a CGW satisfying the above:
STOP
Ask for the real CGW codebase
Do NOT create a placeholder
A1.1 Reference Implementation (Minimum Viable)
If no CGW exists, implement the following minimal structure (acceptable baseline).
It must satisfy routing + FastMem dependency + recurrent workspace update.
Workspace Update Loop (Reference)
def cgw_step(S_t, mem_t, x_t, params, t):
    routing_logits = params.W_route @ S_t
    routing_probs = gumbel_softmax(routing_logits, tau=params.tau)
    selected = int(torch.argmax(routing_probs))

    if selected == 0:  # io
        delta_S = params.io_encoder(x_t)
    elif selected == 1:  # logic
        delta_S = params.logic_mlp(S_t)
    elif selected == 2:  # mem
        key = params.key_proj(S_t)
        value = mem_t.read(key)
        if value is None:
            value = torch.zeros_like(S_t)
        mem_t.write(key, params.val_proj(S_t))
        delta_S = params.mem_merge(S_t, value)
    elif selected == 3:  # sup
        delta_S = params.sup_mlp(S_t)
        halt_logit = params.halt_head(S_t)

    S_next = S_t + delta_S

    step_info = {
        "selected": selected,
        "routing_probs": routing_probs.detach().cpu(),
        "reads": mem_t.read_count_delta(),
        "writes": mem_t.write_count_delta(),
        "halt_logit": float(halt_logit.detach().cpu()) if selected == 3 else None,
    }
    return S_next, mem_t, step_info
FastMem (Reference)
class SimpleFastMem:
    def __init__(self):
        self.store = {}
        self.read_count = 0
        self.write_count = 0
        self._prev_read = 0
        self._prev_write = 0

    def _hash(self, key):
        return hash(key.detach().cpu().numpy().tobytes())

    def write(self, key, value):
        self.store[self._hash(key)] = value.detach().clone()
        self.write_count += 1

    def read(self, key):
        self.read_count += 1
        return self.store.get(self._hash(key), None)

    def read_count_delta(self):
        d = self.read_count - self._prev_read
        self._prev_read = self.read_count
        return d

    def write_count_delta(self):
        d = self.write_count - self._prev_write
        self._prev_write = self.write_count
        return d
B. Dataset (Phase 0)
Implement a toy reasoning dataset with labelled difficulty:
Difficulty	Ratio	Expectation
Easy	60%	should solve most
Medium	30%	partial success
Hard	10%	optional
C. Logging (MANDATORY)
Per Step
routing probabilities
selected specialist
routing entropy (bits)
FastMem reads / writes
workspace norm
halting signal
step index
Per Run
success/failure
trajectory length
per-specialist usage
FastMem read/write ratio
routing entropy stats
D. Phase 0 Threshold Checks (AUTOMATED)
Metric	Red Flag	Context
Routing entropy	< 0.5 bits OR > 1.9 bits	Collapsed (one specialist dominates) OR uniform (no specialization learned). Max possible = log₂(4) = 2.0 bits
FastMem read/write ratio	< 0.1	Memory write-only
Trajectory length variance	std < 0.5×mean	No adaptation
Easy task success	< 20%	Core model broken
Specialist diversity	Fewer than 2 specialists selected ≥10% of steps	Routing collapsed
E. Sanity Tests (MANDATORY)
Sanity Test 1 — Seed determinism
Same input + same seed → identical routing traces
Same input + different seeds → routing traces differ (at least sometimes)
Sanity Test 2 — Specialist activation
≥2 specialists selected ≥10% of steps (aggregated)
Sanity Test 3 — FastMem data dependency (causal proof)
For 3 successful runs:
Identify a key K written at step t₁
Confirm the same key K is read at step t₂ > t₁
Re-run with modified write value at t₁:
Replace value with value + noise, where noise ~ N(0, 0.1·‖value‖)
OR replace value with zeros_like(value)
Keep key, seed, inputs, and routing identical
Verify:
Read value at t₂ changes
Downstream behaviour diverges (workspace, routing, or final output)
This proves the read uses stored content, not a stub.
F. Debug / Visualisation Mode
CLI flags:
--debug-mode
--visualize
Debug mode:
print routing decisions per step
dump FastMem keys read/written
save first 5 trajectories as text
Visualisation:
routing usage plot
routing entropy over time
FastMem read/write heatmap
G. Resource Guardrails
Abort and log if:
any sample > 60s
routing loop repeats 5+ times
memory > 4GB
NaN/Inf in workspace
H. Acceptance Criteria
Phase 0 is DONE only if:
training + evaluation run end-to-end
reports and logs produced
all sanity tests pass
red flags absent or explainable
⭐ Minimum Passing Example (Optional but Recommended)
Task: 3-digit addition (e.g. 123 + 456)
Typical trace:
Step 0: io
Step 1–2: logic
Step 3: mem write carry
Step 4: mem read carry
Step 5: logic
Step 6: sup halt
Metrics:
Routing entropy ≈ 1.2 bits
Specialists: io, logic, mem, sup all used
FastMem: ≥1 read & write
Output correct
Phase 0.5 — Human Review (MANDATORY)
Before Phase 1:
Review phase0_report.md
Inspect 5 success + 5 failure traces
Confirm routing is state-dependent
Confirm FastMem causal usage
Write 1 paragraph:
“I believe K>1 will help because…”
Expected Effort
2–4 days, not weeks.
If stuck after 3 days:
missing task clarity
architectural ambiguity
tooling issues
Do not hyperparameter-tune Phase 0.
Final Instruction to AI Agent
Implement exactly this.
Ask before assuming.
Do not simplify semantics.
