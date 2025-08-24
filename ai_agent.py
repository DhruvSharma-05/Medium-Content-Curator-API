import requests
import json
import pandas as pd
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Article:
    title: str
    author: str
    url: str
    published_date: datetime
    summary: str
    tags: List[str]
    claps: int
    reading_time: int
    engagement_score: float
    content_preview: str
    member_only: bool = False
    publication: str = ""
    author_followers: int = 0

class MediumScraper:
    """Advanced web scraper for Medium articles with rate limiting and user-agent rotation"""
    
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.request_count = 0
        self.last_request_time = 0
        self.min_delay = 1  # Minimum delay between requests
        self.max_delay = 3  # Maximum delay between requests
        
    def _get_headers(self):
        """Get randomized headers to avoid detection"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Longer delay every 10 requests
        if self.request_count % 10 == 0:
            time.sleep(random.uniform(3, 7))
    
    def get_topic_articles(self, topic: str, limit: int = 30) -> List[Dict]:
        """Get articles from Medium topic pages with enhanced scraping"""
        logger.info(f"Scraping articles for topic: {topic}")
        
        # Try multiple approaches for getting topic articles
        articles = []
        
        # Method 1: RSS feeds (most reliable)
        rss_articles = self._get_topic_rss(topic, limit // 2)
        articles.extend(rss_articles)
        
        # Method 2: Topic page scraping
        if len(articles) < limit:
            page_articles = self._scrape_topic_page(topic, limit - len(articles))
            articles.extend(page_articles)
        
        # Method 3: Search-based scraping
        if len(articles) < limit:
            search_articles = self._scrape_search_results(topic, limit - len(articles))
            articles.extend(search_articles)
        
        return self._deduplicate_articles(articles)[:limit]
    
    def _get_topic_rss(self, topic: str, limit: int) -> List[Dict]:
        """Get articles from Medium RSS feeds"""
        topic_feeds = {
            'ai': 'https://medium.com/feed/tag/artificial-intelligence',
            'artificial-intelligence': 'https://medium.com/feed/tag/artificial-intelligence',
            'machine-learning': 'https://medium.com/feed/tag/machine-learning',
            'programming': 'https://medium.com/feed/tag/programming',
            'javascript': 'https://medium.com/feed/tag/javascript',
            'python': 'https://medium.com/feed/tag/python',
            'react': 'https://medium.com/feed/tag/react',
            'data-science': 'https://medium.com/feed/tag/data-science',
            'startup': 'https://medium.com/feed/tag/startup',
            'design': 'https://medium.com/feed/tag/design',
            'technology': 'https://medium.com/feed/tag/technology',
            'blockchain': 'https://medium.com/feed/tag/blockchain',
            'web-development': 'https://medium.com/feed/tag/web-development',
            'mobile': 'https://medium.com/feed/tag/mobile-development'
        }
        
        topic_key = topic.lower().replace(' ', '-')
        feed_url = topic_feeds.get(topic_key, f'https://medium.com/feed/tag/{topic_key}')
        
        try:
            self._rate_limit()
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:limit]:
                article_data = self._parse_rss_entry(entry)
                if article_data:
                    articles.append(article_data)
            
            logger.info(f"Got {len(articles)} articles from RSS for topic: {topic}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS for topic {topic}: {e}")
            return []
    
    def _scrape_topic_page(self, topic: str, limit: int) -> List[Dict]:
        """Scrape Medium topic pages directly"""
        try:
            topic_url = f"https://medium.com/tag/{topic.lower().replace(' ', '-')}"
            self._rate_limit()
            
            response = self.session.get(topic_url, headers=self._get_headers())
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Look for article elements in the page
            article_elements = soup.find_all('article') or soup.find_all('div', class_=re.compile(r'story|article'))
            
            for element in article_elements[:limit]:
                article_data = self._extract_article_from_element(element)
                if article_data:
                    articles.append(article_data)
            
            logger.info(f"Scraped {len(articles)} articles from topic page: {topic}")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping topic page {topic}: {e}")
            return []
    
    def _scrape_search_results(self, query: str, limit: int) -> List[Dict]:
        """Scrape Medium search results"""
        try:
            search_url = f"https://medium.com/search?q={query.replace(' ', '%20')}"
            self._rate_limit()
            
            response = self.session.get(search_url, headers=self._get_headers())
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Extract articles from search results
            for element in soup.find_all('div', class_=re.compile(r'story|search-result'))[:limit]:
                article_data = self._extract_article_from_element(element)
                if article_data:
                    articles.append(article_data)
            
            logger.info(f"Got {len(articles)} articles from search: {query}")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []
    
    def get_trending_articles(self, limit: int = 50) -> List[Dict]:
        """Get trending articles from Medium"""
        articles = []
        
        # Method 1: Main RSS feed
        try:
            self._rate_limit()
            main_feed = feedparser.parse("https://medium.com/feed")
            for entry in main_feed.entries[:limit//2]:
                article_data = self._parse_rss_entry(entry)
                if article_data:
                    articles.append(article_data)
        except Exception as e:
            logger.error(f"Error fetching main feed: {e}")
        
        # Method 2: Today's top stories
        try:
            trending_articles = self._scrape_trending_page(limit - len(articles))
            articles.extend(trending_articles)
        except Exception as e:
            logger.error(f"Error fetching trending: {e}")
        
        return self._deduplicate_articles(articles)[:limit]
    
    def _scrape_trending_page(self, limit: int) -> List[Dict]:
        """Scrape Medium's trending page"""
        try:
            self._rate_limit()
            response = self.session.get("https://medium.com/", headers=self._get_headers())
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Look for trending articles sections
            trending_sections = soup.find_all('div', class_=re.compile(r'trending|popular|recommended'))
            
            for section in trending_sections:
                article_elements = section.find_all('article') or section.find_all('div', class_=re.compile(r'story'))
                
                for element in article_elements[:limit]:
                    article_data = self._extract_article_from_element(element)
                    if article_data and len(articles) < limit:
                        articles.append(article_data)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping trending page: {e}")
            return []
    
    def get_article_details(self, article_url: str) -> Optional[Dict]:
        """Get detailed information about a specific article"""
        try:
            self._rate_limit()
            response = self.session.get(article_url, headers=self._get_headers())
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract detailed article information
            article_data = {
                'url': article_url,
                'title': self._extract_title(soup),
                'author': self._extract_author(soup),
                'published_date': self._extract_publish_date(soup),
                'reading_time': self._extract_reading_time(soup),
                'claps': self._extract_claps(soup),
                'responses': self._extract_responses(soup),
                'content': self._extract_content(soup),
                'tags': self._extract_tags(soup),
                'summary': self._extract_summary(soup),
                'member_only': self._is_member_only(soup),
                'publication': self._extract_publication(soup)
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error getting article details for {article_url}: {e}")
            return None
    
    def _parse_rss_entry(self, entry) -> Optional[Dict]:
        """Parse RSS entry into article data"""
        try:
            # Extract reading time from content or estimate
            content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else entry.get('summary', '')
            reading_time = self._estimate_reading_time(content)
            
            # Extract tags
            tags = [tag.term for tag in entry.get('tags', [])]
            
            # Parse published date
            try:
                published_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else datetime.now()
            except:
                published_date = datetime.now()
            
            # Extract author
            author = entry.get('author', 'Unknown')
            if isinstance(author, dict):
                author = author.get('name', 'Unknown')
            
            return {
                'title': entry.title,
                'author': author,
                'url': entry.link,
                'published_date': published_date,
                'summary': self._clean_html(entry.get('summary', '')),
                'tags': tags,
                'reading_time': reading_time,
                'content_preview': self._clean_html(content)[:300] + '...' if len(content) > 300 else self._clean_html(content),
                'claps': 0,  # Not available in RSS
                'member_only': False,  # Assume not member-only
                'publication': entry.get('source', {}).get('title', '') if isinstance(entry.get('source'), dict) else ''
            }
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    def _extract_article_from_element(self, element) -> Optional[Dict]:
        """Extract article data from a BeautifulSoup element"""
        try:
            # Extract title
            title_elem = element.find('h1') or element.find('h2') or element.find('h3')
            title = title_elem.get_text(strip=True) if title_elem else "Untitled"
            
            # Extract URL
            link_elem = element.find('a', href=True)
            url = link_elem['href'] if link_elem else ""
            if url and not url.startswith('http'):
                url = urljoin("https://medium.com", url)
            
            # Extract author
            author_elem = element.find('a', class_=re.compile(r'author')) or element.find('span', class_=re.compile(r'author'))
            author = author_elem.get_text(strip=True) if author_elem else "Unknown"
            
            # Extract summary
            summary_elem = element.find('p') or element.find('div', class_=re.compile(r'subtitle|summary'))
            summary = summary_elem.get_text(strip=True) if summary_elem else ""
            
            # Extract reading time
            time_elem = element.find('span', string=re.compile(r'min read')) or element.find('span', class_=re.compile(r'read-time'))
            reading_time = self._extract_reading_time_from_text(time_elem.get_text() if time_elem else "5 min read")
            
            # Extract claps (if available)
            claps_elem = element.find('span', class_=re.compile(r'clap')) or element.find('button', class_=re.compile(r'clap'))
            claps = self._extract_number_from_text(claps_elem.get_text() if claps_elem else "0")
            
            if title and url:
                return {
                    'title': title,
                    'author': author,
                    'url': url,
                    'published_date': datetime.now(),  # Would need more sophisticated parsing
                    'summary': summary,
                    'tags': [],  # Would need to be extracted separately
                    'reading_time': reading_time,
                    'content_preview': summary,
                    'claps': claps,
                    'member_only': '★' in element.get_text() or 'member' in element.get_text().lower(),
                    'publication': ""
                }
            
        except Exception as e:
            logger.error(f"Error extracting article from element: {e}")
        
        return None
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on URL"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles
    
    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time based on word count"""
        words = len(re.findall(r'\w+', self._clean_html(content)))
        return max(1, words // 200)  # Average 200 words per minute
    
    def _extract_reading_time_from_text(self, text: str) -> int:
        """Extract reading time from text like '5 min read'"""
        match = re.search(r'(\d+)\s*min', text.lower())
        return int(match.group(1)) if match else 5
    
    def _extract_number_from_text(self, text: str) -> int:
        """Extract number from text (for claps, followers, etc.)"""
        # Handle K, M suffixes
        text = text.strip().lower()
        match = re.search(r'([\d.]+)([km]?)', text)
        if match:
            number, suffix = match.groups()
            number = float(number)
            if suffix == 'k':
                return int(number * 1000)
            elif suffix == 'm':
                return int(number * 1000000)
            return int(number)
        return 0
    
    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags from content"""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(strip=True)
    
    # Additional helper methods for detailed article scraping
    def _extract_title(self, soup) -> str:
        title_elem = soup.find('h1') or soup.find('title')
        return title_elem.get_text(strip=True) if title_elem else "Untitled"
    
    def _extract_author(self, soup) -> str:
        author_elem = soup.find('a', rel='author') or soup.find('span', class_=re.compile(r'author'))
        return author_elem.get_text(strip=True) if author_elem else "Unknown"
    
    def _extract_publish_date(self, soup) -> datetime:
        date_elem = soup.find('time') or soup.find('span', class_=re.compile(r'date'))
        if date_elem:
            date_str = date_elem.get('datetime') or date_elem.get_text()
            try:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                pass
        return datetime.now()
    
    def _extract_reading_time(self, soup) -> int:
        time_elem = soup.find('span', string=re.compile(r'min read'))
        if time_elem:
            return self._extract_reading_time_from_text(time_elem.get_text())
        return 5
    
    def _extract_claps(self, soup) -> int:
        claps_elem = soup.find('span', class_=re.compile(r'clap')) or soup.find('button', class_=re.compile(r'clap'))
        if claps_elem:
            return self._extract_number_from_text(claps_elem.get_text())
        return 0
    
    def _extract_responses(self, soup) -> int:
        responses_elem = soup.find('span', string=re.compile(r'response')) or soup.find('a', string=re.compile(r'response'))
        if responses_elem:
            return self._extract_number_from_text(responses_elem.get_text())
        return 0
    
    def _extract_content(self, soup) -> str:
        content_elem = soup.find('article') or soup.find('div', class_=re.compile(r'content|story'))
        return content_elem.get_text(strip=True) if content_elem else ""
    
    def _extract_tags(self, soup) -> List[str]:
        tag_elements = soup.find_all('a', class_=re.compile(r'tag')) or soup.find_all('span', class_=re.compile(r'tag'))
        return [elem.get_text(strip=True) for elem in tag_elements]
    
    def _extract_summary(self, soup) -> str:
        summary_elem = soup.find('meta', property='og:description') or soup.find('meta', name='description')
        if summary_elem:
            return summary_elem.get('content', '')
        # Fallback to first paragraph
        p_elem = soup.find('p')
        return p_elem.get_text(strip=True) if p_elem else ""
    
    def _is_member_only(self, soup) -> bool:
        return bool(soup.find(string=re.compile(r'member.?only', re.I))) or bool(soup.find('span', string='★'))
    
    def _extract_publication(self, soup) -> str:
        pub_elem = soup.find('a', class_=re.compile(r'publication')) or soup.find('span', class_=re.compile(r'publication'))
        return pub_elem.get_text(strip=True) if pub_elem else ""

class AIArticleSuggester:
    """AI-powered article suggestion engine with advanced scraping"""
    
    def __init__(self):
        self.scraper = MediumScraper()
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        self.articles_cache = []
        self.vectors = None
        self.user_history = {}  # Store user interaction history
    
    def fetch_articles(self, topics: List[str] = None, limit_per_topic: int = 25) -> List[Dict]:
        """Fetch articles from multiple sources with enhanced scraping"""
        logger.info(f"Fetching articles for topics: {topics}")
        all_articles = []
        
        if not topics:
            # Get trending articles if no specific topics
            all_articles = self.scraper.get_trending_articles(limit=50)
        else:
            # Fetch articles for each topic
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self.scraper.get_topic_articles, topic, limit_per_topic) 
                          for topic in topics]
                
                for future in futures:
                    try:
                        articles = future.result(timeout=30)
                        all_articles.extend(articles)
                    except Exception as e:
                        logger.error(f"Error fetching articles: {e}")
        
        # Remove duplicates and cache
        unique_articles = self.scraper._deduplicate_articles(all_articles)
        self.articles_cache = unique_articles
        logger.info(f"Cached {len(unique_articles)} unique articles")
        
        return unique_articles
    
    def calculate_engagement_score(self, article: Dict) -> float:
        """Calculate AI engagement score with enhanced factors"""
        score = 0.0
        
        # Recency score (30 points max)
        if 'published_date' in article and article['published_date']:
            days_old = (datetime.now() - article['published_date']).days
            recency_score = max(0, (30 - days_old) / 30) * 30
            score += recency_score
        
        # Reading time optimization (20 points max)
        reading_time = article.get('reading_time', 5)
        if 4 <= reading_time <= 12:
            time_score = 20 - abs(reading_time - 8) * 2
            score += max(0, time_score)
        else:
            score += 5
        
        # Title quality (15 points max)
        title = article.get('title', '')
        title_words = len(title.split())
        if 5 <= title_words <= 15:
            score += 15
        else:
            score += max(0, 15 - abs(title_words - 10))
        
        # Content quality (20 points max)
        summary = article.get('summary', '')
        content_preview = article.get('content_preview', '')
        content_length = len(summary) + len(content_preview)
        if content_length > 100:
            score += min(20, content_length / 50)
        
        # Claps/engagement (10 points max)
        claps = article.get('claps', 0)
        if claps > 0:
            score += min(10, np.log10(claps + 1) * 3)
        
        # Author credibility (5 points max)
        if article.get('author', 'Unknown') != 'Unknown':
            score += 5
        
        # Publication bonus (5 points max)
        if article.get('publication', ''):
            score += 5
        
        # Non-member content bonus (5 points max)
        if not article.get('member_only', False):
            score += 5
        
        return min(100, score)
    
    def get_personalized_suggestions(
        self, 
        user_preferences: Dict,
        search_query: str = "",
        limit: int = 10,
        user_id: str = None
    ) -> List[Dict]:
        """Get AI-powered personalized suggestions"""
        
        # Fetch fresh articles
        topics = user_preferences.get('topics', [])
        articles = self.fetch_articles(topics, limit_per_topic=30)
        
        if not articles:
            logger.warning("No articles found")
            return []
        
        # Build content vectors
        self.build_content_vectors(articles)
        
        # Score each article
        scored_articles = []
        for i, article in enumerate(articles):
            try:
                # Base engagement score
                base_score = self.calculate_engagement_score(article)
                
                # User preference matching
                preference_score = self._calculate_preference_match(article, user_preferences)
                
                # Search query relevance
                query_score = self._calculate_query_relevance(article, search_query)
                
                # Diversity bonus (avoid too similar articles)
                diversity_score = self._calculate_diversity_bonus(article, scored_articles)
                
                # User history bonus (if available)
                history_score = self._calculate_history_match(article, user_id) if user_id else 0
                
                # Final AI score
                final_score = base_score + preference_score + query_score + diversity_score + history_score
                
                article_with_score = article.copy()
                article_with_score['ai_score'] = round(final_score, 2)
                article_with_score['engagement_score'] = round(base_score, 2)
                article_with_score['relevance_breakdown'] = {
                    'engagement': round(base_score, 2),
                    'preferences': round(preference_score, 2),
                    'query_match': round(query_score, 2),
                    'diversity': round(diversity_score, 2),
                    'history': round(history_score, 2)
                }
                
                scored_articles.append(article_with_score)
                
            except Exception as e:
                logger.error(f"Error scoring article: {e}")
                continue
        
        # Sort by AI score and apply diversity filter
        scored_articles.sort(key=lambda x: x['ai_score'], reverse=True)
        diverse_articles = self._apply_diversity_filter(scored_articles, limit * 2)
        
        return diverse_articles[:limit]
    
    def _calculate_preference_match(self, article: Dict, preferences: Dict) -> float:
        """Enhanced preference matching"""
        score = 0.0
        
        # Reading time preference (25 points max)
        reading_time_pref = preferences.get('reading_time', 'any')
        article_time = article.get('reading_time', 5)
        
        if reading_time_pref == 'short' and article_time <= 5:
            score += 25
        elif reading_time_pref == 'medium' and 5 < article_time <= 12:
            score += 25
        elif reading_time_pref == 'long' and article_time > 12:
            score += 25
        elif reading_time_pref == 'any':
            score += 15
        
        # Topic preference matching (30 points max)
        preferred_topics = [topic.lower() for topic in preferences.get('topics', [])]
        article_tags = [tag.lower() for tag in article.get('tags', [])]
        article_title = article.get('title', '').lower()
        article_content = (article.get('summary', '') + ' ' + article.get('content_preview', '')).lower()
        
        topic_matches = 0
        for topic in preferred_topics:
            # Direct tag match (highest weight)
            if any(topic in tag for tag in article_tags):
                topic_matches += 3
            # Title match (medium weight)
            elif topic in article_title:
                topic_matches += 2
            # Content match (lower weight)
            elif topic in article_content:
                topic_matches += 1
        
        score += min(30, topic_matches * 5)
        
        # Recency preference (15 points max)
        recency_pref = preferences.get('recency', 'week')
        if 'published_date' in article and article['published_date']:
            days_old = (datetime.now() - article['published_date']).days
            
            if recency_pref == 'day' and days_old <= 1:
                score += 15
            elif recency_pref == 'week' and days_old <= 7:
                score += 15
            elif recency_pref == 'month' and days_old <= 30:
                score += 15
            elif recency_pref == 'any':
                score += 10
        
        # Difficulty preference (10 points max)
        difficulty_pref = preferences.get('difficulty', 'any')
        if difficulty_pref != 'any':
            # Estimate difficulty based on reading time, content complexity
            estimated_difficulty = self._estimate_article_difficulty(article)
            if estimated_difficulty == difficulty_pref:
                score += 10
            elif abs(self._difficulty_to_number(estimated_difficulty) - self._difficulty_to_number(difficulty_pref)) == 1:
                score += 5
        else:
            score += 5
        
        return score
    
    def _calculate_query_relevance(self, article: Dict, query: str) -> float:
        """Enhanced search query relevance calculation"""
        if not query:
            return 0
        
        query_lower = query.lower()
        query_words = query_lower.split()
        score = 0.0
        
        # Title relevance (40 points max)
        title = article.get('title', '').lower()
        for word in query_words:
            if word in title:
                score += 10
        
        # Exact title match bonus
        if query_lower in title:
            score += 20
        
        # Summary relevance (25 points max)
        summary = article.get('summary', '').lower()
        summary_matches = sum(1 for word in query_words if word in summary)
        score += min(25, summary_matches * 5)
        
        # Tags relevance (20 points max)
        tags = [tag.lower() for tag in article.get('tags', [])]
        for word in query_words:
            for tag in tags:
                if word in tag:
                    score += 10
                    break
        
        # Author relevance (10 points max)
        author = article.get('author', '').lower()
        if any(word in author for word in query_words):
            score += 10
        
        return min(95, score)
    
    def _calculate_diversity_bonus(self, article: Dict, existing_articles: List[Dict]) -> float:
        """Calculate diversity bonus to avoid too similar articles"""
        if not existing_articles:
            return 5
        
        # Check topic diversity
        article_tags = set(tag.lower() for tag in article.get('tags', []))
        existing_tags = set()
        for existing in existing_articles[-5:]:  # Check last 5 articles
            existing_tags.update(tag.lower() for tag in existing.get('tags', []))
        
        # Bonus for introducing new topics
        new_topics = len(article_tags - existing_tags)
        diversity_score = min(10, new_topics * 2)
        
        # Check author diversity
        article_author = article.get('author', '')
        existing_authors = [existing.get('author', '') for existing in existing_articles[-3:]]
        if article_author not in existing_authors and article_author != 'Unknown':
            diversity_score += 5
        
        return diversity_score
    
    def _calculate_history_match(self, article: Dict, user_id: str) -> float:
        """Calculate score based on user reading history"""
        if user_id not in self.user_history:
            return 0
        
        user_data = self.user_history[user_id]
        score = 0.0
        
        # Preferred topics based on history
        preferred_topics = user_data.get('preferred_topics', {})
        article_tags = article.get('tags', [])
        
        for tag in article_tags:
            if tag.lower() in preferred_topics:
                score += preferred_topics[tag.lower()] * 10
        
        # Preferred reading times
        preferred_time = user_data.get('avg_reading_time', 0)
        article_time = article.get('reading_time', 5)
        if preferred_time > 0:
            time_diff = abs(article_time - preferred_time)
            score += max(0, 10 - time_diff)
        
        return min(20, score)
    
    def _apply_diversity_filter(self, articles: List[Dict], limit: int) -> List[Dict]:
        """Apply diversity filter to ensure varied suggestions"""
        if len(articles) <= limit:
            return articles
        
        selected = []
        used_authors = set()
        used_topics = set()
        
        for article in articles:
            author = article.get('author', '')
            topics = set(tag.lower() for tag in article.get('tags', []))
            
            # Diversity rules
            author_diverse = author not in used_authors or len(used_authors) < 3
            topic_diverse = len(topics & used_topics) < 3 or len(selected) < limit // 2
            
            if (author_diverse and topic_diverse) or len(selected) < limit // 3:
                selected.append(article)
                used_authors.add(author)
                used_topics.update(topics)
                
                if len(selected) >= limit:
                    break
        
        # Fill remaining slots if needed
        remaining = limit - len(selected)
        if remaining > 0:
            for article in articles:
                if article not in selected and remaining > 0:
                    selected.append(article)
                    remaining -= 1
        
        return selected
    
    def _estimate_article_difficulty(self, article: Dict) -> str:
        """Estimate article difficulty based on content"""
        reading_time = article.get('reading_time', 5)
        title = article.get('title', '')
        content = article.get('summary', '') + ' ' + article.get('content_preview', '')
        
        # Technical terms that indicate advanced content
        advanced_terms = ['algorithm', 'neural', 'optimization', 'architecture', 'implementation', 
                         'framework', 'methodology', 'systematic', 'comprehensive']
        
        intermediate_terms = ['tutorial', 'guide', 'learn', 'understanding', 'basics', 
                             'introduction', 'beginner']
        
        # Count technical terms
        advanced_count = sum(1 for term in advanced_terms if term in content.lower())
        intermediate_count = sum(1 for term in intermediate_terms if term in content.lower())
        
        # Decision logic
        if advanced_count >= 3 or reading_time > 15:
            return 'advanced'
        elif intermediate_count >= 2 or 'beginner' in title.lower():
            return 'beginner'
        else:
            return 'intermediate'
    
    def _difficulty_to_number(self, difficulty: str) -> int:
        """Convert difficulty to number for comparison"""
        mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        return mapping.get(difficulty, 2)
    
    def build_content_vectors(self, articles: List[Dict]):
        """Build TF-IDF vectors for content similarity"""
        if not articles:
            return
        
        documents = []
        for article in articles:
            # Combine multiple text fields for better vectorization
            text_parts = [
                article.get('title', ''),
                article.get('summary', ''),
                ' '.join(article.get('tags', [])),
                article.get('content_preview', '')
            ]
            text = ' '.join(filter(None, text_parts))
            documents.append(text)
        
        try:
            self.vectors = self.vectorizer.fit_transform(documents)
            logger.info(f"Built vectors for {len(documents)} articles")
        except Exception as e:
            logger.error(f"Error building vectors: {e}")
            self.vectors = None
    
    def get_similar_articles(self, article_index: int, limit: int = 5) -> List[Dict]:
        """Get articles similar to a specific article"""
        if not self.vectors or article_index >= self.vectors.shape[0]:
            return []
        
        try:
            article_vector = self.vectors[article_index]
            similarities = cosine_similarity(article_vector, self.vectors).flatten()
            
            # Get most similar articles (excluding the article itself)
            similar_indices = similarities.argsort()[::-1][1:limit+1]
            
            similar_articles = []
            for idx in similar_indices:
                if idx < len(self.articles_cache):
                    article = self.articles_cache[idx].copy()
                    article['similarity_score'] = round(similarities[idx], 3)
                    similar_articles.append(article)
            
            return similar_articles
        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return []
    
    def update_user_history(self, user_id: str, article: Dict, action: str):
        """Update user interaction history for better recommendations"""
        if user_id not in self.user_history:
            self.user_history[user_id] = {
                'preferred_topics': {},
                'reading_times': [],
                'liked_articles': [],
                'disliked_articles': []
            }
        
        user_data = self.user_history[user_id]
        
        if action == 'read' or action == 'like':
            # Update preferred topics
            for tag in article.get('tags', []):
                topic = tag.lower()
                user_data['preferred_topics'][topic] = user_data['preferred_topics'].get(topic, 0) + 0.1
            
            # Update reading time preferences
            reading_time = article.get('reading_time', 5)
            user_data['reading_times'].append(reading_time)
            
            # Keep only recent reading times
            if len(user_data['reading_times']) > 50:
                user_data['reading_times'] = user_data['reading_times'][-50:]
            
            # Calculate average reading time
            user_data['avg_reading_time'] = sum(user_data['reading_times']) / len(user_data['reading_times'])
            
        if action == 'like':
            user_data['liked_articles'].append(article.get('url', ''))
        elif action == 'dislike':
            user_data['disliked_articles'].append(article.get('url', ''))

# Main AI Agent class
class MediumAIAgent:
    """Advanced agentic AI for Medium article suggestions with web scraping"""
    
    def __init__(self):
        self.suggester = AIArticleSuggester()
        self.cache_duration = timedelta(hours=2)  # Cache articles for 2 hours
        self.last_cache_update = {}
    
    def suggest_articles(
        self,
        user_preferences: Dict,
        search_query: str = "",
        limit: int = 10,
        user_id: str = None,
        force_refresh: bool = False
    ) -> Dict:
        """Main method to get intelligent article suggestions"""
        try:
            # Check if we need to refresh cache
            topics_key = ','.join(sorted(user_preferences.get('topics', [])))
            should_refresh = (
                force_refresh or 
                topics_key not in self.last_cache_update or
                datetime.now() - self.last_cache_update[topics_key] > self.cache_duration
            )
            
            if should_refresh:
                logger.info("Refreshing article cache...")
                self.last_cache_update[topics_key] = datetime.now()
            
            suggestions = self.suggester.get_personalized_suggestions(
                user_preferences, search_query, limit, user_id
            )
            
            # Add metadata
            for suggestion in suggestions:
                suggestion['scraped_at'] = datetime.now().isoformat()
                suggestion['cache_key'] = topics_key
            
            return {
                'success': True,
                'suggestions': suggestions,
                'total_count': len(suggestions),
                'timestamp': datetime.now().isoformat(),
                'cache_status': 'refreshed' if should_refresh else 'cached',
                'search_query': search_query,
                'user_preferences': user_preferences
            }
        
        except Exception as e:
            logger.error(f"Error in suggest_articles: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'suggestions': [],
                'total_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trending_articles(self, limit: int = 20) -> Dict:
        """Get trending articles with AI scoring"""
        try:
            articles = self.suggester.scraper.get_trending_articles(limit * 2)
            
            # Score articles
            for article in articles:
                article['engagement_score'] = self.suggester.calculate_engagement_score(article)
                article['ai_score'] = article['engagement_score']  # Use engagement as AI score for trending
            
            # Sort by engagement score
            articles.sort(key=lambda x: x['engagement_score'], reverse=True)
            
            return {
                'success': True,
                'trending_articles': articles[:limit],
                'total_count': len(articles[:limit]),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting trending articles: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'trending_articles': [],
                'total_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def search_articles(self, query: str, limit: int = 20) -> Dict:
        """Search for articles with AI ranking"""
        try:
            articles = self.suggester.scraper._scrape_search_results(query, limit * 2)
            
            # Score articles based on search relevance
            for article in articles:
                base_score = self.suggester.calculate_engagement_score(article)
                query_score = self.suggester._calculate_query_relevance(article, query)
                article['ai_score'] = base_score + query_score
                article['query_relevance'] = query_score
            
            # Sort by AI score
            articles.sort(key=lambda x: x['ai_score'], reverse=True)
            
            return {
                'success': True,
                'search_results': articles[:limit],
                'total_count': len(articles[:limit]),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'search_results': [],
                'total_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_article_details(self, article_url: str) -> Dict:
        """Get detailed information about a specific article"""
        try:
            article_details = self.suggester.scraper.get_article_details(article_url)
            
            if article_details:
                article_details['engagement_score'] = self.suggester.calculate_engagement_score(article_details)
                
                return {
                    'success': True,
                    'article': article_details,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Article not found or could not be scraped',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error getting article details: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def track_user_interaction(self, user_id: str, article_url: str, action: str) -> Dict:
        """Track user interaction for better recommendations"""
        try:
            # Find the article in cache
            article = None
            for cached_article in self.suggester.articles_cache:
                if cached_article.get('url') == article_url:
                    article = cached_article
                    break
            
            if article:
                self.suggester.update_user_history(user_id, article, action)
                
                return {
                    'success': True,
                    'message': f'Tracked {action} for user {user_id}',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Article not found in cache',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error tracking interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI agent
    agent = MediumAIAgent()
    
    # Test user preferences
    user_prefs = {
        'topics': ['AI', 'Machine Learning', 'Python'],
        'reading_time': 'medium',
        'recency': 'week',
        'difficulty': 'intermediate'
    }
    
    print("🤖 Testing Medium AI Agent with Web Scraping...")
    print("=" * 60)
    
    # Test suggestions
    print("Getting AI-powered suggestions...")
    result = agent.suggest_articles(
        user_preferences=user_prefs,
        search_query="artificial intelligence",
        limit=5,
        user_id="test_user"
    )
    
    if result['success']:
        print(f"\n✅ Found {result['total_count']} suggestions:")
        for i, article in enumerate(result['suggestions'], 1):
            print(f"\n{i}. {article['title'][:60]}...")
            print(f"   Author: {article.get('author', 'Unknown')}")
            print(f"   AI Score: {article.get('ai_score', 0):.1f}")
            print(f"   Reading Time: {article.get('reading_time', 0)} min")
            print(f"   Tags: {', '.join(article.get('tags', [])[:3])}")
            if 'relevance_breakdown' in article:
                breakdown = article['relevance_breakdown']
                print(f"   Scoring: Eng={breakdown['engagement']:.1f}, Pref={breakdown['preferences']:.1f}, Query={breakdown['query_match']:.1f}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test trending
    print("\n" + "=" * 60)
    print("Getting trending articles...")
    trending_result = agent.get_trending_articles(limit=3)
    
    if trending_result['success']:
        print(f"✅ Found {trending_result['total_count']} trending articles:")
        for i, article in enumerate(trending_result['trending_articles'], 1):
            print(f"\n{i}. {article['title'][:60]}...")
            print(f"   Engagement Score: {article.get('engagement_score', 0):.1f}")
    else:
        print(f"❌ Trending Error: {trending_result['error']}")