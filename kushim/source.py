import wikipedia
import datetime
import os
from typing import List, Dict, Any, Optional, Protocol, Literal, TypedDict
from llama_index.core import SimpleDirectoryReader

# Protocol Definition for Extensible Sources
# This new section introduces a protocol-based architecture for data sourcing.
# By defining a standard `Source` interface, we decouple the pipeline from any
# specific data provider (like Wikipedia). Now, any class that adheres to the
# `Source` protocol can be used, making the entire system extensible.

class SourceDocument(TypedDict):
    """A standardized dictionary to represent a single document from any source."""
    title: str
    content: str
    metadata: Dict[str, Any]

class Source(Protocol):
    """
    A protocol that defines the standard interface for all data sources.
    Any class that implements a `fetch` method matching this signature
    can be used by the Kushim pipeline.
    """
    def fetch(self, **kwargs) -> List[SourceDocument]:
        """
        Fetches documents from the source based on provided arguments.
        
        Returns:
            A list of SourceDocument objects.
        """
        ...

# Wikipedia Source Implementation
# The `WikipediaSource` is refactored to be an implementation of the `Source`
# protocol. Its public interface is now a single `fetch` method, which
# provides a consistent entry point while still supporting all the previous
# discovery modes (search, geosearch, etc.).

class WikipediaSource:
    """
    A source extractor for fetching and curating content from Wikipedia.
    It implements the Source protocol, making it pluggable into the pipeline.
    """
    def __init__(self, user_agent: str = None):
        """
        Initializes the WikipediaSource.
        """
        user_agent = user_agent or "Kushim/0.3.0 (https://github.com/teilomillet/kushim; teilomillet@gmail.com) Research-tutorial"
        wikipedia.set_user_agent(user_agent)
        wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(milliseconds=50))

    def fetch(
        self,
        mode: Literal['article', 'search', 'geosearch', 'random', 'linked'] = 'article',
        **kwargs
    ) -> List[SourceDocument]:
        """
        Main entry point for fetching articles from Wikipedia. Dispatches to the
        appropriate internal method based on the 'mode'.
        """
        if mode == 'article':
            return self._fetch_single_article(**kwargs)
        elif mode == 'search':
            return self._search_and_filter_articles(**kwargs)
        elif mode == 'geosearch':
            return self._geosearch_and_filter_articles(**kwargs)
        elif mode == 'random':
            return self._get_random_articles(**kwargs)
        elif mode == 'linked':
            return self._get_linked_articles(**kwargs)
        else:
            raise ValueError(f"Unknown Wikipedia fetch mode: {mode}")

    def _to_source_documents(self, articles: List[Dict[str, Any]]) -> List[SourceDocument]:
        """Converts the rich article dictionary from the wikipedia library into SourceDocument objects."""
        return [
            SourceDocument(
                title=art['title'],
                content=art['content'],
                metadata={k: v for k, v in art.items() if k not in ['title', 'content']}
            )
            for art in articles
        ]

    def _get_page(self, title: str) -> Optional[wikipedia.WikipediaPage]:
        """
        A robust internal helper to fetch a WikipediaPage object.
        
        Handles common errors like PageNotFound and Disambiguation pages,
        allowing higher-level methods to proceed smoothly.
        """
        try:
            # Preload ensures all content is fetched, reducing subsequent API calls.
            return wikipedia.page(title, auto_suggest=False, preload=True)
        except wikipedia.exceptions.PageError:
            print(f"Warning: Article '{title}' not found. Skipping.")
            return None
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Warning: '{title}' is a disambiguation page. Options: {e.options}. Skipping.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching '{title}': {e}. Skipping.")
            return None

    def _fetch_single_article(self, article_title: str) -> List[SourceDocument]:
        page = self._get_page(article_title)
        if not page:
            return []
        
        article_dict = {
            "title": page.title,
            "content": page.content,
            "summary": page.summary,
            "links": page.links,
        }
        return self._to_source_documents([article_dict])

    def _search_and_filter_articles(
        self,
        query: str,
        search_results_to_consider: int = 20,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[SourceDocument]:
        """
        Searches for articles, filters them by word count, and returns the longest ones.

        This is a powerful discovery method for finding the most substantial articles
        on a given topic, which are often the best sources for dataset generation.

        Returns:
            A list of dictionaries, where each dictionary represents an article
            and contains its title, content, word count, summary, and links.
            The list is sorted by word count in descending order.
        """
        print(f"Searching for the top {num_articles_to_return} articles for query: '{query}'...")
        article_titles = wikipedia.search(query, results=search_results_to_consider)
        
        print(f"Found {len(article_titles)} potential articles. Fetching and filtering...")
        pages = [self._get_page(title) for title in article_titles if title]
        filtered_articles = self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )
        return self._to_source_documents(filtered_articles)

    def _geosearch_and_filter_articles(
        self,
        latitude: float,
        longitude: float,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
        radius: int = 10000,
        search_results_to_consider: int = 20,
    ) -> List[SourceDocument]:
        """
        Finds articles via geosearch, filters by word count, and returns the longest.
        Ideal for creating location-based datasets.
        """
        print(f"Geosearching for articles near ({latitude}, {longitude})...")
        article_titles = wikipedia.geosearch(
            latitude,
            longitude,
            results=search_results_to_consider,
            radius=radius
        )
        
        print(f"Found {len(article_titles)} potential articles. Fetching and filtering...")
        pages = [self._get_page(title) for title in article_titles if title]
        filtered_articles = self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )
        return self._to_source_documents(filtered_articles)

    def _get_random_articles(
        self,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[SourceDocument]:
        """
        Fetches random articles, filters them by word count, and returns the longest.
        Excellent for creating diverse, unbiased datasets for general knowledge.
        """
        print(f"Fetching {num_articles_to_return} random articles...")
        # The API is limited to 10 random pages at a time.
        # We may need to make multiple calls if more are requested.
        article_titles = set()
        while len(article_titles) < num_articles_to_return:
            needed = num_articles_to_return - len(article_titles)
            # The wikipedia.random `pages` parameter has a maximum of 10
            pages_to_fetch = min(needed, 10)
            try:
                random_titles = wikipedia.random(pages=pages_to_fetch)
                # It can sometimes return a single string instead of a list
                if isinstance(random_titles, str):
                    article_titles.add(random_titles)
                else:
                    article_titles.update(random_titles)
            except Exception as e:
                print(f"An error occurred while fetching random articles: {e}")
                break

        print(f"Found {len(article_titles)} potential articles. Fetching and filtering...")
        pages = [self._get_page(title) for title in article_titles if title]
        filtered_articles = self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )
        return self._to_source_documents(filtered_articles)

    def _get_linked_articles(
        self,
        article_title: str,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[SourceDocument]:
        """
        Finds articles linked from a seed article, filters by word count, and returns the longest.
        Powerful for creating interconnected, domain-specific datasets.
        """
        print(f"Fetching articles linked from '{article_title}'...")
        page = self._get_page(article_title)
        if not page:
            return []

        linked_titles = page.links
        print(f"Found {len(linked_titles)} linked articles. Fetching and filtering...")
        
        pages = [self._get_page(title) for title in linked_titles if title]
        filtered_articles = self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )
        return self._to_source_documents(filtered_articles)
    
    def _filter_and_sort_pages(
        self,
        pages: List[wikipedia.WikipediaPage],
        num_articles_to_return: int,
        min_word_count: int,
    ) -> List[Dict[str, Any]]:
        articles_with_meta = []
        for page in pages:
            if page and page.content:
                word_count = len(page.content.split())
                if word_count >= min_word_count:
                    articles_with_meta.append({
                        "title": page.title,
                        "content": page.content,
                        "word_count": word_count,
                        "summary": page.summary,
                        "links": page.links,
                    })
        
        sorted_articles = sorted(
            articles_with_meta, key=lambda x: x['word_count'], reverse=True
        )
        
        print(f"Found {len(sorted_articles)} articles meeting the minimum word count of {min_word_count}.")
        final_articles = sorted_articles[:num_articles_to_return]
        print(f"Returning the top {len(final_articles)} articles.")
        return final_articles

# Local File Source Implementation
# This new class is an example of the extensibility enabled by the `Source`
# protocol. It allows the pipeline to read from local files, a
# feature that was impossible with the previous rigid design. It leverages
# llama-index to automatically handle various file types.

class LocalFileSource:
    """
    A source extractor for fetching content from local files using llama-index.
    It implements the Source protocol, allowing it to be used interchangeably
    with other sources in the pipeline. It supports various file types like
    .md, .pdf, .docx, .json, etc., automatically selecting the appropriate loader.
    """
    def fetch(self, path: str) -> List[SourceDocument]:
        """
        Fetches content from a single file or all supported files in a directory.
        
        Args:
            path: A path to a single file or a directory.
        """
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            return []

        if os.path.isfile(path):
            # For a single file, we pass it as a list to the reader
            documents = SimpleDirectoryReader(input_files=[path]).load_data()
        elif os.path.isdir(path):
            # For a directory, it reads all supported files
            documents = SimpleDirectoryReader(input_dir=path).load_data()
        else:
            return []
        
        # Convert llama-index documents to SourceDocument format
        source_documents = []
        for doc in documents:
            title = doc.metadata.get("file_name", os.path.basename(doc.metadata.get("file_path", "unknown")))
            source_documents.append(
                SourceDocument(
                    title=title,
                    content=doc.text,
                    metadata=doc.metadata
                )
            )
        
        return source_documents
