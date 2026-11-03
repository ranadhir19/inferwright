"""llm_mesh.workflows: end-to-end mesh pipelines from the concept doc.

Two canonical mesh workflows:

1. ``fraud_claim_pipeline`` -- "Find fraud risk in these insurance
   claims": vision extraction -> reasoning risk analysis -> tool agent
   (database check) -> chat summary. Sequential with dependencies.
2. ``chart_report_pipeline`` -- "Explain this chart and summarize the
   report": vision (chart) and chat (report) fan out in parallel, then a
   reasoning model combines the answer.
"""

from .orchestrator import Orchestrator, Subtask


def fraud_claim_pipeline(orch: Orchestrator, rng, request_id: str,
                         trace=None):
    """Sequential 4-stage fraud-risk pipeline with a tool-agent stage."""
    subtasks = [
        Subtask("extract-claims", "vision", in_tokens=900, out_tokens=300,
                prompt="Extract structured fields from claim documents."),
        Subtask("risk-reasoning", "reasoning", in_tokens=700, out_tokens=250,
                prompt="Score fraud risk from extracted claim fields."),
        Subtask("db-check", "tool", in_tokens=100, out_tokens=50,
                prompt="Cross-check claimant against fraud database.",
                tool="fraud-db"),
        Subtask("final-summary", "chat", in_tokens=600, out_tokens=200,
                prompt="Summarize fraud findings for the adjuster."),
    ]
    return orch.run(rng, request_id, subtasks, mode="sequential", trace=trace)


def chart_report_pipeline(orch: Orchestrator, rng, request_id: str,
                          trace=None):
    """Parallel fan-out (vision + chat) then a reasoning combine step."""
    fanout = [
        Subtask("interpret-chart", "vision", in_tokens=800, out_tokens=250,
                prompt="Interpret the chart: trends, outliers, axes."),
        Subtask("summarize-report", "chat", in_tokens=1500, out_tokens=300,
                prompt="Summarize the quarterly report text."),
    ]
    part1 = orch.run(rng, request_id + ":fanout", fanout, mode="parallel",
                     trace=trace)
    combine = [
        Subtask("combine-answer", "reasoning", in_tokens=900, out_tokens=250,
                prompt="Combine chart interpretation and report summary "
                       "into one coherent answer."),
    ]
    part2 = orch.run(rng, request_id + ":combine", combine,
                     mode="sequential", trace=trace)
    # stitch the two stages into one result
    records = part1.records + part2.records
    return type(part1)(
        request_id=request_id, mode="parallel+sequential", records=records,
        wall_latency_ms=part1.wall_latency_ms + part2.wall_latency_ms,
        total_cost_usd=part1.total_cost_usd + part2.total_cost_usd,
        mean_quality=sum(r.quality for r in records) / len(records),
        n_failovers=part1.n_failovers + part2.n_failovers,
    )
