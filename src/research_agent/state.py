import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from research_agent.search import Source


class Finding(TypedDict):
    sub_question: str
    sources: list[Source]


class Plan(BaseModel):
    sub_questions: list[str] = Field(description="3 to 5 focused, searchable sub-questions")


class Critique(BaseModel):
    all_sub_questions_answered: bool = Field(description="Does every sub-question have its own answered section?")
    score: int = Field(ge=1, le=10, description="Overall quality from 1 to 10")
    verdict: Literal["pass", "rewrite", "research"] = Field(
        description="pass if good; rewrite if the writing needs work; research if facts are missing"
    )
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix")
    follow_up_questions: list[str] = Field(
        default_factory=list, description="New searchable questions, only when verdict is research"
    )


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[str]
    pending: list[str]  # sub-questions to research in the current round
    findings: Annotated[list[Finding], operator.add]  # merged across parallel researchers
    draft: str
    critique: dict
    passed: bool
    loop_again: bool  # critic decided another revision loop is allowed and needed
    revisions: int
    report_path: str


class ResearchTask(TypedDict):
    sub_question: str
