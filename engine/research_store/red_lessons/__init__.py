"""engine.research_store.red_lessons — structured RED Lesson schema + store.

This package upgrades RED records from "event marked failed, don't retry"
into a learning object that captures:

  - WHICH failure mode (F1..F9, controlled taxonomy)
  - WHICH mechanism family (taxonomy; e.g. carry, momentum, attention)
  - paper anchors (motivation + critique, DOI + section + quote)
  - subsumption map (which deployed factor kills this one)
  - forward research vectors (what's worth trying next, given THIS failure)
  - dormant_revisit conditions (new data / market / tool → re-test)

Doctrine (2026-06-03):

  1. Lessons are SEPARATE from events (engine.research_store.schema).
     An event is "what happened"; a lesson is "what we learned and how
     to use it going forward". Events stay immutable per the 2026-06-02
     research-store doctrine; lessons can be amended (with version
     bumps + parent_lesson_id chain).
  2. Failure modes use the F1..F9 controlled vocabulary (see
     `failure_modes.py`). Free-form failure descriptions are NOT
     allowed in the failure_modes list — they go in failure_evidence
     strings.
  3. Mechanism family uses the controlled vocabulary
     (see `mechanism_families.py`). No fuzzy auto-mapping.
  4. Paper anchors are first-class. A lesson without a motivation paper
     is fine (some REDs come from internal hypothesis), but if cited
     it must include DOI + year + key_claim.

API surface:

    from engine.research_store.red_lessons import (
        REDLesson, FailureMode, MechanismFamily,
        PaperRef, ForwardVector, ReviewState,
        load_lessons, save_lesson,
    )
"""
from engine.research_store.red_lessons.failure_modes import (
    FailureMode,
    FAILURE_MODE_DOCS,
)
from engine.research_store.red_lessons.mechanism_families import (
    MechanismFamily,
    MECHANISM_FAMILY_DOCS,
)
from engine.research_store.red_lessons.schema import (
    REDLesson,
    PaperRef,
    ForwardVector,
    DormantRevisit,
    ReviewState,
    LessonStrength,
    GroundingMethod,
    LESSON_SCHEMA_VERSION,
    PRETRAIN_GROUNDED_FREEZE_TS,
)

# VerbatimQuote re-export — public API. Lazy via __getattr__ to avoid
# cycle at package init (hypothesis.schema imports red_lessons.mechanism_families,
# which triggers this __init__ to run; we cannot eager-import hypothesis
# back at that moment).
def __getattr__(name):
    if name == "VerbatimQuote":
        from engine.research_store.hypothesis.schema import VerbatimQuote
        return VerbatimQuote
    raise AttributeError(f"module 'engine.research_store.red_lessons' has no attribute {name!r}")
from engine.research_store.red_lessons.store import (
    load_lessons,
    save_lesson,
    LESSONS_PATH,
)

__all__ = [
    "FailureMode", "FAILURE_MODE_DOCS",
    "MechanismFamily", "MECHANISM_FAMILY_DOCS",
    "REDLesson", "PaperRef", "ForwardVector", "DormantRevisit",
    "ReviewState", "LessonStrength", "LESSON_SCHEMA_VERSION",
    "GroundingMethod", "VerbatimQuote", "PRETRAIN_GROUNDED_FREEZE_TS",
    "load_lessons", "save_lesson", "LESSONS_PATH",
]
