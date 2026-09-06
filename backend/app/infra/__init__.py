"""Cross-cutting plumbing: the single ffmpeg subprocess wrapper and workspace
path management.

Carries no domain *rules* - nothing here knows what an overlay is or when to
remove one. It does use domain data shapes (`VideoMeta`), because defining that
shape twice to preserve a zero-import rule would trade a real duplication for a
cosmetic one.
"""
