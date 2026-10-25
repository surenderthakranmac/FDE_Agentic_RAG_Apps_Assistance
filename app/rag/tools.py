# Isolates the network configuration interfaces and outbound external search APIs.
from app.core.config import settings
from app.core.logger_setup import logger


class InternetSearchTool:
    """
    Abstracted Search Interface. Routes out-of-bounds queries
    directly through the official Tavily SDK using your secure client key.
    """

    @staticmethod
    def execute(query: str, max_results: int = 2) -> str:
        logger.info(
            "Executing outbound search query over live channels",
            extra={"telemetry": {"query": query}},
        )
        try:
            from tavily import TavilyClient

            tavily_client = TavilyClient(api_key=settings.tavily_api_key)
            response = tavily_client.search(query=query, search_depth="advanced")
            results = response.get("results", [])
            if not results:
                return "No search answers returned from public web channels."

            formatted_results = []
            for i in range(min(max_results, len(results))):
                res = results[i]
                formatted_results.append(
                    f"[Web Source {i+1}]: {res.get('title')}\n"
                    f"URL: {res.get('url')}\n"
                    f"Context: {res.get('content')}\n"
                )
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Outbound search engine interface crash exception: {str(e)}")
            return (
                f"Error executing internet fallback request layer via Tavily: {str(e)}"
            )
