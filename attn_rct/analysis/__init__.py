"""Demsar (2006) analysis pipeline for the attention RCT.

Stages, each a module:
    synth       generate synthetic results in train.py's schema, for testing
    collect     results/*.json -> tidy long table, with completeness + integrity checks
    aggregate   seeds -> cells; within-design ranking with direction handling

Later stages (omnibus, posthoc, effects, plots, run) build on these.
"""
