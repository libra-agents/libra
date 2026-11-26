"""ToolReAGt agentic tool retrieval with LlamaIndex ReAct."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from agentic_retrieval.backends import SearchBackend, SearchHit
from agentic_retrieval.clients import (
    ModelOutputError,
    OpenAIChatClient,
    TokenUsage,
)


CHECKPOINTS = (1, 3, 5, 7, 10)
RETRIEVER_TOOL_NAME = "tool_retriever"
PROMPT_CONDITIONS = (
    "plain_fulltask",
    "subtask",
    "subtask_fulltask",
    "fulltask_decomp_single",
    "fulltask_decomp_multi",
)

# Appendix B of Braunschweiler et al. (2025), with only whitespace normalized.
TOOLREAGT_REACT_SYSTEM_HEADER = """\
You are designed to help with a variety of tasks, from answering questions to
providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using the
tools in any sequence you deem appropriate to complete the task at hand. This
may require breaking the task into sub-tasks and using different tools to
complete each sub-task.

You have access to the following tools:
{tool_desc}

## Output Format
Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a
tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs
(e.g. {{"input": "hello world", "num_beams": 5}})

Please ALWAYS start with a Thought. NEVER surround your response with markdown
code markers. You may use code markers within your response if you need to.
Please use a valid JSON format for the Action Input.
Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

Observation: tool response

You should keep repeating the above format till you have enough information to
answer the question without using any more tools. At that point, you MUST respond
in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language
to answer
Answer: [your answer here, in the same language as the user's question]

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here, in the same language as the user's question]

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant
messages.
"""


def format_appendix_a_prompt(
    *,
    condition: str,
    task: str,
    subtask: str | None = None,
    decomposition: list[dict[str, Any]] | None = None,
) -> str:
    """Render one of the input-information conditions defined in Appendix A."""

    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown Appendix A prompt condition: {condition}")
    if condition == "plain_fulltask":
        return (
            f'Given the following complex task: "{task}" and the list of tools '
            "in the context, select the best tools to solve the complex task. "
            f'You must obtain that context by calling "{RETRIEVER_TOOL_NAME}" '
            "with a suitable query before selecting tools."
        )
    if subtask is None:
        raise ValueError(f"{condition} requires an official subtask.")
    if condition == "subtask":
        return (
            f'Given the following task: "{subtask}", select the best tool '
            "provided in the context to solve the task. Obtain the context by "
            f'calling "{RETRIEVER_TOOL_NAME}" with a suitable query.'
        )
    if condition == "subtask_fulltask":
        return (
            f'Given the following complex task: "{task}", select the best tool '
            f'provided in the context to solve the following substep: "{subtask}". '
            f'Obtain the context by calling "{RETRIEVER_TOOL_NAME}".'
        )
    if decomposition is None:
        raise ValueError(f"{condition} requires an official decomposition.")
    decomposition_text = json.dumps(decomposition, ensure_ascii=False)
    cardinality = (
        "best tool"
        if condition == "fulltask_decomp_single"
        else "best tools, ranked by relevance"
    )
    return (
        "You are an expert in selecting tools to solve a given task. "
        f'The complex task is: "{task}" and its decomposition into sub-tasks is: '
        f"{decomposition_text}. Given this context, and the list of tools "
        f'provided by calling "{RETRIEVER_TOOL_NAME}" with a suitable query, '
        f'select the {cardinality} to solve the following substep: "{subtask}".'
    )


@dataclass(frozen=True)
class RetrievalCall:
    """One successful call to the retriever meta-tool."""

    call: int
    query: str
    ranking: list[str]
    hits: list[dict[str, Any]]
    model_output: str
    request_token_usage: dict[str, int]
    cumulative_token_usage: dict[str, int]
    embedding_requests_cumulative: int
    reranker_requests_cumulative: int


@dataclass(frozen=True)
class GenerationTurn:
    """One raw ReAct generation, retained for reproducibility and auditing."""

    request: int
    content: str
    model_finish_reason: str
    token_usage: dict[str, int]
    cumulative_token_usage: dict[str, int]


@dataclass(frozen=True)
class RetrievalCheckpoint:
    """Latest retrieval state available at an evaluation checkpoint."""

    ranking: list[str]
    source_call: int | None
    cumulative_token_usage: dict[str, int]
    embedding_requests_cumulative: int
    reranker_requests_cumulative: int


@dataclass
class RetrievalTrajectory:
    task: str
    seed: int
    max_retrieval_calls: int
    generation_turns: list[GenerationTurn]
    retrieval_calls: list[RetrievalCall]
    checkpoints: dict[str, RetrievalCheckpoint]
    final_answer: str | None
    finish_reason: str
    token_usage: dict[str, int]
    generation_requests: int
    embedding_requests: int
    reranker_requests: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_checkpoints(
    calls: list[RetrievalCall],
    *,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> dict[str, RetrievalCheckpoint]:
    """Take the latest call at each t, forward-filling after natural stopping."""

    output: dict[str, RetrievalCheckpoint] = {}
    zero_usage = TokenUsage().to_dict()
    for checkpoint in checkpoints:
        available = [call for call in calls if call.call <= checkpoint]
        latest = available[-1] if available else None
        output[str(checkpoint)] = RetrievalCheckpoint(
            ranking=[] if latest is None else list(latest.ranking),
            source_call=None if latest is None else latest.call,
            cumulative_token_usage=(
                zero_usage
                if latest is None
                else dict(latest.cumulative_token_usage)
            ),
            embedding_requests_cumulative=(
                0 if latest is None else latest.embedding_requests_cumulative
            ),
            reranker_requests_cumulative=(
                0 if latest is None else latest.reranker_requests_cumulative
            ),
        )
    return output


class ToolReAGtRetriever:
    """Retrieval-only ToolReAGt agent.

    ReAct formatting and parsing are imported from the pinned LlamaIndex source.
    Only the retriever meta-tool is exposed; benchmark domain tools are never
    executed.
    """

    def __init__(
        self,
        *,
        backend: SearchBackend,
        chat_client: OpenAIChatClient,
        max_retrieval_calls: int = 10,
        prompt_condition: str = "plain_fulltask",
    ) -> None:
        if max_retrieval_calls != 10:
            raise ValueError("The curve protocol uses one max-10 trajectory.")
        self.backend = backend
        self.chat_client = chat_client
        self.max_retrieval_calls = max_retrieval_calls
        if prompt_condition not in PROMPT_CONDITIONS:
            raise ValueError(f"Unknown Appendix A prompt condition: {prompt_condition}")
        self.prompt_condition = prompt_condition
        self.formatter = ReActChatFormatter.from_defaults(
            system_header=TOOLREAGT_REACT_SYSTEM_HEADER
        )
        self.parser = ReActOutputParser()

        def search_tools(input: str) -> str:
            """Retrieve an ordered list of candidate tools for a focused query."""

            raise RuntimeError("The runner records and executes this tool directly.")

        self.tool = FunctionTool.from_defaults(
            fn=search_tools,
            name=RETRIEVER_TOOL_NAME,
            description=backend.tool_description,
        )

    @staticmethod
    def _messages(
        formatter: ReActChatFormatter,
        tool: FunctionTool,
        task: str,
        reasoning: list[BaseReasoningStep],
        prompt_condition: str,
        subtask: str | None,
        decomposition: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        user_prompt = format_appendix_a_prompt(
            condition=prompt_condition,
            task=task,
            subtask=subtask,
            decomposition=decomposition,
        )
        formatted = formatter.format(
            tools=[tool],
            chat_history=[ChatMessage(role=MessageRole.USER, content=user_prompt)],
            current_reasoning=reasoning,
        )
        return [
            {"role": message.role.value, "content": str(message.content or "")}
            for message in formatted
        ]

    @staticmethod
    def _observation(hits: list[SearchHit]) -> str:
        return json.dumps(
            [hit.observation() for hit in hits],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def run(
        self,
        task: str,
        *,
        seed: int,
        subtask: str | None = None,
        decomposition: list[dict[str, Any]] | None = None,
    ) -> RetrievalTrajectory:
        reasoning: list[BaseReasoningStep] = []
        calls: list[RetrievalCall] = []
        generation_turns: list[GenerationTurn] = []
        usage = TokenUsage()
        generation_requests = 0
        final_answer: str | None = None
        finish_reason = "max_retrieval_calls"

        while len(calls) < self.max_retrieval_calls:
            try:
                completion = self.chat_client.complete(
                    self._messages(
                        self.formatter,
                        self.tool,
                        task,
                        reasoning,
                        self.prompt_condition,
                        subtask,
                        decomposition,
                    ),
                    seed=seed,
                )
                generation_requests += 1
                usage = usage + completion.usage
                generation_turns.append(
                    GenerationTurn(
                        request=generation_requests,
                        content=completion.content,
                        model_finish_reason=completion.finish_reason,
                        token_usage=completion.usage.to_dict(),
                        cumulative_token_usage=usage.to_dict(),
                    )
                )
                step = self.parser.parse(completion.content)
            except (ModelOutputError, RuntimeError, ValueError) as error:
                finish_reason = f"invalid_model_output: {error}"
                break

            if isinstance(step, ResponseReasoningStep):
                if not calls:
                    finish_reason = "mandatory_retriever_not_called"
                else:
                    final_answer = step.response
                    finish_reason = "natural_answer"
                break
            if not isinstance(step, ActionReasoningStep):
                finish_reason = f"unsupported_reasoning_step: {type(step).__name__}"
                break
            if step.action != RETRIEVER_TOOL_NAME:
                finish_reason = f"unknown_action: {step.action}"
                break
            query = str(step.action_input.get("input", "")).strip()
            if not query:
                finish_reason = "missing_retrieval_input"
                break

            try:
                hits = self.backend.search(query)
            except (OSError, RuntimeError, ValueError) as error:
                finish_reason = f"retriever_error: {error}"
                break
            reasoning.extend(
                [
                    step,
                    ObservationReasoningStep(observation=self._observation(hits)),
                ]
            )
            call_index = len(calls) + 1
            calls.append(
                RetrievalCall(
                    call=call_index,
                    query=query,
                    ranking=[hit.name for hit in hits],
                    hits=[hit.observation() for hit in hits],
                    model_output=completion.content,
                    request_token_usage=completion.usage.to_dict(),
                    cumulative_token_usage=usage.to_dict(),
                    embedding_requests_cumulative=(
                        call_index
                        * self.backend.embedding_requests_per_search
                    ),
                    reranker_requests_cumulative=(
                        call_index
                        * self.backend.reranker_requests_per_search
                    ),
                )
            )

        return RetrievalTrajectory(
            task=task,
            seed=seed,
            max_retrieval_calls=self.max_retrieval_calls,
            generation_turns=generation_turns,
            retrieval_calls=calls,
            checkpoints=build_checkpoints(calls),
            final_answer=final_answer,
            finish_reason=finish_reason,
            token_usage=usage.to_dict(),
            generation_requests=generation_requests,
            embedding_requests=(
                len(calls) * self.backend.embedding_requests_per_search
            ),
            reranker_requests=(
                len(calls) * self.backend.reranker_requests_per_search
            ),
        )
