from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class KushimConfig:
    """
    A centralized configuration object for the Kushim pipeline.

    This dataclass holds all the settings that control the behavior of the
    pipeline, from data sourcing and chunking to generation and validation.
    Using a single configuration object makes the pipeline easier to manage,
    extend, and reason about.
    """
    # Model Configuration
    model_name: str = 'groq/llama3-8b-8192'
    llm: Optional[Any] = None # Allows passing a pre-configured dspy.LM

    # Parallelism Configuration
    max_workers: int = 4

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Q&A Generation Configuration
    question_style: str = "narrative"
    num_questions_per_chunk: int = 1
    
    # Source Configuration
    # These are not direct settings but are passed through to the source's
    # fetch method, so we define a place for them here.
    fetch_kwargs: Dict[str, Any] = field(default_factory=dict) 