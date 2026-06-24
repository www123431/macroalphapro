"""engine.research_store.forward_vectors — paper-driven forward direction surface.

T5 of the locked PAPER → HYPOTHESIS → TEST → VERDICT chain
(2026-06-04). For each ingested paper, we know all its extracted
Hypothesis records (T3). For each REDLesson, we know which
hypothesis_ids it tested (T1+T2+T4). The DIFFERENCE — hypotheses NOT
yet cited by any lesson — is the genuine forward research surface.

This is NOT the old `red_lessons.schema.ForwardVector` dataclass
(which was a pretrain-speculation guess inside the lesson). This is a
real, paper-grounded "what's testable that hasn't been tested" surface.

Public API:

    from engine.research_store.forward_vectors import (
        ForwardVectorV2,
        generate_forward_vectors,
        FORWARD_VECTORS_PATH,
        load_forward_vectors, save_forward_vector,
    )
"""
from engine.research_store.forward_vectors.schema import (
    FORWARD_VECTOR_SCHEMA_VERSION,
    ForwardVectorV2,
    Priority,
)
from engine.research_store.forward_vectors.generator import (
    generate_forward_vectors,
)
from engine.research_store.forward_vectors.store import (
    FORWARD_VECTORS_PATH,
    load_forward_vectors,
    save_forward_vector,
)

__all__ = [
    "ForwardVectorV2", "Priority",
    "FORWARD_VECTOR_SCHEMA_VERSION",
    "generate_forward_vectors",
    "FORWARD_VECTORS_PATH",
    "load_forward_vectors", "save_forward_vector",
]
