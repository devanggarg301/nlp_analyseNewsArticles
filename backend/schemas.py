from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from .article_dates import publication_from_text

class WorkspaceInput(BaseModel):
    title: str = Field(min_length=3,max_length=160)

class ArticleInput(BaseModel):
    title: str = Field(min_length=1,max_length=300)
    publisher: str = Field(min_length=1,max_length=150)
    text: str = Field(min_length=30,max_length=60000)
    url: str | None = Field(default=None,max_length=2000)
    published_at: str | None = Field(default=None,max_length=100)
    source_type: Literal["paste","url","upload","demo"] = "paste"
    @model_validator(mode="after")
    def infer_publication_date(self):
        if not self.published_at:
            self.published_at = publication_from_text(self.text)
        return self
    @field_validator("text")
    @classmethod
    def word_limit(cls, v):
        if len(v.split()) > 3000:
            raise ValueError("Each article must contain at most 3,000 words.")
        return v.replace("\r\n","\n")
    @field_validator("url")
    @classmethod
    def safe_source_url(cls, v):
        if v and not v.startswith(("http://","https://")):
            raise ValueError("Source URL must use HTTP or HTTPS.")
        return v

class URLInput(BaseModel):
    url: str = Field(min_length=8,max_length=2000)

class AnalyzeInput(BaseModel):
    confirm_unrelated: bool = False

class ChatInput(BaseModel):
    question: str = Field(min_length=3,max_length=1000)

class MergeInput(BaseModel):
    names: list[str] = Field(min_length=2,max_length=10)
    canonical: str = Field(min_length=1,max_length=100)

