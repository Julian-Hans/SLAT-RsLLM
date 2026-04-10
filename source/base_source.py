"""
BaseSource: constructs prompt representations for search engines.

Reconstructed from the prompt templates in fine_tune_flan/train.py
and the usage pattern in source_selection.py.
"""

SYSTEM_INSTRUCTION = (
    "Federated search retrieves information from a variety of sources via a search application "
    "built on top of one or more search engines. A user makes a single query request. The federated "
    "search then selects only the search engines that the query should be sent to from a list of "
    "search engines, and aggregates the result for presentation of high quality result to the user. "
    "The task is called resource selection."
)

TASK_INSTRUCTION = (
    "Now, please reply only yes or no to indicate if the query should be sent to the search engine.\n"
    "Response:"
)

TASK_INSTRUCTION_SNIPPET = (
    "Now, please reply only yes or no to indicate if the user query should be sent to the search engine.\n"
    "Response:"
)

PROMPT_NAME = (
    SYSTEM_INSTRUCTION +
    "The following is a search engine with its name and url\n\n"
    "Name: {name}\n"
    "URL: {url}\n"
    "The following is a real user query: \n"
    "{query}\n" +
    TASK_INSTRUCTION
)

PROMPT_NAME_DESCRIPTION = (
    SYSTEM_INSTRUCTION +
    "The following is a search engine with its name, url and description\n\n"
    "Name: {name}\n"
    "URL: {url}\n"
    "Description: {description}\n"
    "The following is a real user query: \n"
    "{query}\n" +
    TASK_INSTRUCTION
)

PROMPT_NAME_SNIPPET = (
    SYSTEM_INSTRUCTION +
    "The following is a search engine with its name and url\n\n"
    "Name: {name}\n"
    "URL: {url}\n"
    "The following is a real user query: \n"
    "{query}\n"
    "The following are some snippets from this search engine that are similar to the user query: \n"
    "{snippet}\n" +
    TASK_INSTRUCTION_SNIPPET
)

PROMPT_EXAMPLE_SNIPPET = (
    SYSTEM_INSTRUCTION +
    "The following is a search engine with its name, url and description\n\n"
    "Name: {name}\n"
    "URL: {url}\n"
    "Description: {description}\n"
    "The following is a real user query: \n"
    "{query}\n"
    "The following are some snippets from this search engine that are similar to the user query: \n"
    "{snippet}\n" +
    TASK_INSTRUCTION_SNIPPET
)

PROMPT_BLIND_DESCRIPTION = (
    SYSTEM_INSTRUCTION +
    "The following is a search engine described only by its content:\n\n"
    "Description: {description}\n"
    "The following is a real user query: \n"
    "{query}\n" +
    TASK_INSTRUCTION
)

REPRESENTATION_PROMPTS = {
    "name": PROMPT_NAME,
    "name_description": PROMPT_NAME_DESCRIPTION,
    "name_snippet": PROMPT_NAME_SNIPPET,
    "example_snippet": PROMPT_EXAMPLE_SNIPPET,
    "blind_description": PROMPT_BLIND_DESCRIPTION,
}


class BaseSource:
    """Represents a search engine resource for ReSLLM prompting."""

    def __init__(self, source_dict):
        self.name = source_dict.get("name", source_dict.get("engineID", ""))
        self.url = source_dict.get("url", "")
        self.description = source_dict.get("description", "")

    def get_representation(self, source_representation, llm_type=None):
        """
        Build a prompt template string for this source.

        The returned string contains a {query} placeholder (and optionally
        {snippet}) that must be filled in at inference time.

        Args:
            source_representation: One of "name", "name_description",
                "name_snippet", "example_snippet".
            llm_type: Unused, kept for API compatibility.

        Returns:
            A prompt string with {query} placeholder.
        """
        prompt_template = REPRESENTATION_PROMPTS.get(source_representation)
        if prompt_template is None:
            raise ValueError(
                f"Unknown source representation: {source_representation!r}. "
                f"Choose from: {list(REPRESENTATION_PROMPTS.keys())}"
            )

        # Pre-fill the source-specific fields, leave {query} (and {snippet}) for later
        return prompt_template.format(
            name=self.name,
            url=self.url,
            description=self.description,
            query="{query}",
            snippet="{snippet}" if "snippet" in prompt_template else "",
        )
