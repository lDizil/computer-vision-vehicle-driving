from hybrid_drive.safety.arbiter import ArbitrationDecision, SafetyArbiter

from .context import SlowContext
from .worker import AsyncSlowWorker, SlowCompletion


class SlowPlanningSession:
    """Explicit plan activation; the first fast decision is final for that plan.

    Waiting for deliberation, if desired in offline experiments, belongs outside
    this session's decision path. Late replies remain audit data, not revisions
    to a trajectory already returned to the caller.
    """

    def __init__(self, arbiter: SafetyArbiter, worker: AsyncSlowWorker):
        if not isinstance(arbiter, SafetyArbiter):
            raise ValueError("arbiter must be SafetyArbiter")
        self.arbiter = arbiter
        self.worker = worker
        self.current_context: SlowContext | None = None
        self.last_completion: SlowCompletion | None = None
        self._matching_completion: SlowCompletion | None = None
        self._decision: ArbitrationDecision | None = None
        self._closed = False

    def activate(self, context: SlowContext) -> str:
        if self._closed:
            raise RuntimeError("Session is closed")
        if not isinstance(context, SlowContext):
            raise ValueError("context must be SlowContext")
        if (self.current_context is not None
                and context.request.request_id == self.current_context.request.request_id):
            raise ValueError("Activation requires a new request_id")
        # Consume queued diagnostics before replacing the active association.
        self._poll()
        disposition = self.worker.submit(context)
        self.current_context = context
        self._matching_completion = None
        self._decision = None
        return disposition

    def decide(self, *, now_sim_s: float) -> ArbitrationDecision:
        if self._closed:
            raise RuntimeError("Session is closed")
        if self.current_context is None:
            raise RuntimeError("Activate a context before deciding")
        self._poll()
        if self._decision is not None:
            return self._decision
        context = self.current_context
        completion = self._matching_completion
        self._decision = self.arbiter.decide(
            context.request.plan, context.motion,
            now_monotonic_s=self.worker.now_monotonic_s(), now_sim_s=now_sim_s,
            request=context.request,
            result=None if completion is None else completion.result,
            received_at_monotonic_s=(None if completion is None
                                     else completion.received_at_monotonic_s),
        )
        return self._decision

    def _poll(self):
        for completion in self.worker.poll():
            self.last_completion = completion
            context = self.current_context
            if (context is None or self._matching_completion is not None
                    or completion.outcome != "completed"):
                continue
            incoming = completion.context
            if (incoming.request.request_id != context.request.request_id
                    or incoming.content_hash != context.content_hash
                    or incoming.request.model_version != context.request.model_version):
                continue
            if completion.result is None:
                continue
            try:
                completion.result.validate_for(context.request)
            except ValueError:
                continue
            self._matching_completion = completion

    def close(self) -> None:
        self._closed = True
        self.current_context = None
        self._matching_completion = None
        self._decision = None
        self.worker.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
