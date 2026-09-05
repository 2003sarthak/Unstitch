"""Orchestration layer: the pipeline, the job runner, the tracking algorithm.

Depends on `domain` Protocols, never on `adapters`. That is what lets the whole
pipeline run against the vision stub with no API key.
"""
