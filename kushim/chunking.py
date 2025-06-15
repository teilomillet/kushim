import typing as t
from llama_index.core.node_parser import SentenceSplitter

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> t.List[str]:
    """
    Breaks a long text into smaller, overlapping chunks while respecting
    sentence boundaries.

    This implementation uses the SentenceSplitter from llama-index to create
    more coherent, semantically meaningful chunks compared to basic word- or
    character-based splitting.

    Args:
        text: The input text to be chunked.
        chunk_size: The target size of each chunk (in tokens).
        overlap: The number of tokens to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    # Instantiate the splitter with the desired configuration.
    # The SentenceSplitter is intelligent enough to split by sentences,
    # then paragraphs, and then by other separators as a fallback.
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    # The 'split_text' method returns a list of text nodes.
    # We extract the text content from each node.
    text_chunks = splitter.split_text(text)
    
    return text_chunks
