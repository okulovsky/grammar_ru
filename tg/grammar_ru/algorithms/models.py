from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class TokenData:
    """Represents data for a single token/word in the NLP processing."""
    text: str
    type: str
    error: bool = False
    suggestions: List[str] = field(default_factory=list)
    hint: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class DataBundle:
    """Represents a bundle of data as it would be passed through the NLP algorithms."""
    tokens: List[TokenData]
    extra_data: Any = None  # This can be used to store additional data like PyMorphy features, etc.
