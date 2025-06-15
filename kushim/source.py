import wikipedia
import datetime
from typing import List, Dict, Any, Optional

class WikipediaSource:
    """
    A source extractor for fetching and curating content from Wikipedia.
    
    This class provides methods to not only fetch single articles but also
    to search for articles on a topic, filter them by length, and return
    a curated list of the most substantial pages, making it a powerful
    tool for sourcing high-quality dataset material.
    """
    def __init__(self, user_agent: str = None):
        """
        Initializes the WikipediaSource.
        
        Sets a user-agent for API requests and enables rate limiting to ensure
        robust and respectful communication with Wikipedia's servers.
        """
        user_agent = user_agent or "Kushim/0.3.0 (https://github.com/teilomillet/kushim; teilomillet@gmail.com) Research-tutorial"
        wikipedia.set_user_agent(user_agent)
        
        # Enable rate limiting to prevent HTTP timeout errors when making many requests.
        wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(milliseconds=50))

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

    def fetch_article_content(self, article_title: str) -> str:
        """
        Fetches the full plain text content of a single Wikipedia article.
        """
        page = self._get_page(article_title)
        return page.content if page else ""

    def search(self, query: str, max_results: int = 20) -> List[str]:
        """
        Performs a Wikipedia search and returns a list of article titles.
        """
        return wikipedia.search(query, results=max_results)

    def _filter_and_sort_pages(
        self,
        pages: List[wikipedia.WikipediaPage],
        num_articles_to_return: int,
        min_word_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Internal helper to filter a list of WikipediaPage objects by word count
        and return the longest ones with their metadata.
        """
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

    def search_and_filter_articles(
        self,
        query: str,
        search_results_to_consider: int = 20,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[Dict[str, Any]]:
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
        article_titles = self.search(query, max_results=search_results_to_consider)
        
        print(f"Found {len(article_titles)} potential articles. Fetching and filtering...")
        pages = [self._get_page(title) for title in article_titles if title]
        return self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )

    def geosearch_and_filter_articles(
        self,
        latitude: float,
        longitude: float,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
        radius: int = 10000,
        search_results_to_consider: int = 20,
    ) -> List[Dict[str, Any]]:
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
        return self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )

    def get_random_articles(
        self,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[Dict[str, Any]]:
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
        return self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )

    def get_linked_articles(
        self,
        article_title: str,
        num_articles_to_return: int = 5,
        min_word_count: int = 1000,
    ) -> List[Dict[str, Any]]:
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
        return self._filter_and_sort_pages(
            [p for p in pages if p],
            num_articles_to_return=num_articles_to_return,
            min_word_count=min_word_count,
        )
