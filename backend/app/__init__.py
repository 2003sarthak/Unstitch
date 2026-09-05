"""Unstitch backend: AI video de-editing.

Layering (dependencies point inward only):

    api      -> services -> domain
    adapters -> domain
    infra    -> domain  (types only, never behaviour)

`services/pipeline.py` imports Protocols from `domain/ports.py` and never a
vendor SDK; `dependencies.py` is the single place where a Protocol is bound to a
concrete adapter.
"""
