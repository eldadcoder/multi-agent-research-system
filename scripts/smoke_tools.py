from research_system.config.settings import get_settings
from research_system.tools.citations import score_reliability
from research_system.tools.errors import ToolError
from research_system.tools.scraper import PageScraper
from research_system.tools.search import get_search_tool

tool = get_search_tool(get_settings())
print("Search backend:", type(tool).__name__)

results = tool.search("solar microgrids rural electrification East Africa", max_results=5)
for r in results:
    print(f"- {r.title}\n  {r.url}  (reliability {score_reliability(r.url):.1f})")

scraper = PageScraper()
for r in results:
    try:
        page = scraper.fetch(r.url)
    except ToolError as e:
        print("skipped:", r.url, "->", e)
        continue
    print("\nFetched:", page.title)
    print(f"{len(page.text)} characters. Preview:\n{page.text[:400]}")
    break