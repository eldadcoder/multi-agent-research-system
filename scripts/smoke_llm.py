from research_system.core.llm import OllamaClient
from research_system.core.models import ResearchPlan

client = OllamaClient()

missing = client.missing_models()
if missing:
    raise SystemExit(
        "Missing models. Run:\n" + "\n".join(f"  ollama pull {m}" for m in missing)
    )

plan = client.structured(
    "You are a research planner.",
    "Question: How do solar microgrids affect rural electrification in East Africa?",
    ResearchPlan,
)
print(plan.model_dump_json(indent=2))
print(client.usage)