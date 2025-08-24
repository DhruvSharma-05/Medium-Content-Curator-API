from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import logging
from ai_agent import MediumAIAgent
import uuid
from typing import List, Dict

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the AI agent (no API token needed for web scraping)
ai_agent = MediumAIAgent()

# In-memory session storage (use Redis/database in production)
user_sessions = {}

# Mock data for testing when scraping fails
MOCK_ARTICLES = [
    {
        'id': '1',
        'title': 'The Future of Artificial Intelligence in 2024',
        'author': 'Sarah Chen',
        'url': 'https://medium.com/@sarahchen/ai-future-2024',
        'published_date': '2024-01-15T10:00:00Z',
        'summary': 'Explore the latest developments in AI technology and what to expect in the coming year.',
        'tags': ['AI', 'Technology', 'Machine Learning', 'Future'],
        'reading_time': 8,
        'ai_score': 9.2,
        'engagement_score': 8.7,
        'claps': 1250,
        'member_only': False,
        'publication': 'Tech Insights'
    },
    {
        'id': '2',
        'title': 'Building Scalable Web Applications with Python',
        'author': 'Alex Rodriguez',
        'url': 'https://medium.com/@alexrodriguez/python-web-apps',
        'published_date': '2024-01-14T14:30:00Z',
        'summary': 'Learn the best practices for creating robust and scalable web applications using Python frameworks.',
        'tags': ['Python', 'Web Development', 'Programming', 'Backend'],
        'reading_time': 12,
        'ai_score': 8.9,
        'engagement_score': 8.4,
        'claps': 890,
        'member_only': False,
        'publication': 'Code Masters'
    },
    {
        'id': '3',
        'title': 'Machine Learning for Beginners: A Complete Guide',
        'author': 'Dr. Emily Watson',
        'url': 'https://medium.com/@emilywatson/ml-beginners-guide',
        'published_date': '2024-01-13T09:15:00Z',
        'summary': 'Start your journey into machine learning with this comprehensive beginner-friendly guide.',
        'tags': ['Machine Learning', 'AI', 'Data Science', 'Beginner'],
        'reading_time': 15,
        'ai_score': 9.5,
        'engagement_score': 9.1,
        'claps': 2100,
        'member_only': False,
        'publication': 'Data Science Weekly'
    },
    {
        'id': '4',
        'title': 'React vs Vue: Which Framework Should You Choose?',
        'author': 'Mike Johnson',
        'url': 'https://medium.com/@mikejohnson/react-vs-vue',
        'published_date': '2024-01-12T16:45:00Z',
        'summary': 'A detailed comparison of React and Vue.js to help you choose the right framework for your project.',
        'tags': ['React', 'Vue.js', 'JavaScript', 'Frontend'],
        'reading_time': 10,
        'ai_score': 8.6,
        'engagement_score': 8.2,
        'claps': 750,
        'member_only': False,
        'publication': 'Web Dev Hub'
    },
    {
        'id': '5',
        'title': 'The Rise of Blockchain Technology in Finance',
        'author': 'Lisa Thompson',
        'url': 'https://medium.com/@lisathompson/blockchain-finance',
        'published_date': '2024-01-11T11:20:00Z',
        'summary': 'How blockchain is revolutionizing the financial industry and what it means for the future.',
        'tags': ['Blockchain', 'Finance', 'Cryptocurrency', 'Innovation'],
        'reading_time': 11,
        'ai_score': 8.8,
        'engagement_score': 8.5,
        'claps': 1100,
        'member_only': False,
        'publication': 'Finance Forward'
    }
]

def get_mock_search_results(query: str, limit: int) -> List[Dict]:
    """Generate mock search results based on query"""
    query_lower = query.lower()
    results = []
    
    for article in MOCK_ARTICLES:
        # Simple relevance scoring
        relevance = 0
        if query_lower in article['title'].lower():
            relevance += 3
        if query_lower in article['summary'].lower():
            relevance += 2
        for tag in article['tags']:
            if query_lower in tag.lower():
                relevance += 1
        
        if relevance > 0:
            article_copy = article.copy()
            article_copy['query_relevance'] = relevance
            results.append(article_copy)
    
    # Sort by relevance and limit
    results.sort(key=lambda x: x['query_relevance'], reverse=True)
    return results[:limit]

def get_mock_trending_results(limit: int) -> List[Dict]:
    """Generate mock trending results"""
    # Return articles sorted by engagement score
    sorted_articles = sorted(MOCK_ARTICLES, key=lambda x: x['engagement_score'], reverse=True)
    return sorted_articles[:limit]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Medium AI Content Curator (Web Scraping)',
        'scraping_status': 'active',
        'features': [
            'AI-powered article suggestions',
            'Web scraping from Medium',
            'Personalized recommendations',
            'Trending articles',
            'Article search',
            'User interaction tracking'
        ]
    })

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """
    Get AI-powered article suggestions using web scraping
    
    Expected JSON payload:
    {
        "user_preferences": {
            "topics": ["AI", "Programming"],
            "reading_time": "medium",
            "recency": "week", 
            "difficulty": "intermediate"
        },
        "search_query": "machine learning",
        "limit": 10,
        "user_id": "optional_user_id",
        "force_refresh": false
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract parameters with defaults
        user_preferences = data.get('user_preferences', {})
        search_query = data.get('search_query', '')
        limit = min(data.get('limit', 10), 50)  # Cap at 50 articles
        user_id = data.get('user_id') or str(uuid.uuid4())
        force_refresh = data.get('force_refresh', False)
        
        # Validate user preferences
        if not isinstance(user_preferences, dict):
            return jsonify({
                'success': False,
                'error': 'user_preferences must be a dictionary'
            }), 400
        
        # Validate topics list
        topics = user_preferences.get('topics', [])
        if not isinstance(topics, list):
            return jsonify({
                'success': False,
                'error': 'topics must be a list'
            }), 400
        
        logger.info(f"Getting suggestions for user {user_id[:8]}... with topics: {topics}")
        
        # Get suggestions from AI agent
        result = ai_agent.suggest_articles(
            user_preferences=user_preferences,
            search_query=search_query,
            limit=limit,
            user_id=user_id,
            force_refresh=force_refresh
        )
        
        # Format response for frontend
        if result['success']:
            formatted_suggestions = []
            for article in result['suggestions']:
                formatted_article = {
                    'id': str(uuid.uuid4()),
                    'title': article.get('title', 'Untitled'),
                    'author': article.get('author', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', datetime.now()).isoformat() if isinstance(article.get('published_date'), datetime) else str(article.get('published_date', '')),
                    'summary': article.get('summary', ''),
                    'tags': article.get('tags', []),
                    'reading_time': article.get('reading_time', 5),
                    'ai_score': round(article.get('ai_score', 0), 1),
                    'engagement_score': round(article.get('engagement_score', 0), 1),
                    'content_preview': article.get('content_preview', ''),
                    'claps': article.get('claps', 0),
                    'member_only': article.get('member_only', False),
                    'publication': article.get('publication', ''),
                    'relevance_breakdown': article.get('relevance_breakdown', {}),
                    'similarity_score': article.get('similarity_score')
                }
                formatted_suggestions.append(formatted_article)
            
            # Store user session
            user_sessions[user_id] = {
                'last_request': datetime.now().isoformat(),
                'preferences': user_preferences,
                'suggestions': formatted_suggestions
            }
            
            return jsonify({
                'success': True,
                'suggestions': formatted_suggestions,
                'total_count': len(formatted_suggestions),
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'cache_status': result.get('cache_status', 'unknown'),
                'search_query': search_query,
                'user_preferences': user_preferences,
                'processing_time': result.get('processing_time')
            })
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in get_suggestions: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Internal error: {str(e)}",
            'suggestions': [],
            'total_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/trending', methods=['GET'])
def get_trending_articles():
    """Get trending articles from Medium using web scraping"""
    try:
        limit = min(request.args.get('limit', 20, type=int), 50)
        
        logger.info(f"Fetching {limit} trending articles...")
        
        # Get trending articles from AI agent
        result = ai_agent.get_trending_articles(limit)
        
        if result['success']:
            formatted_articles = []
            for article in result['trending_articles']:
                formatted_article = {
                    'id': str(uuid.uuid4()),
                    'title': article.get('title', 'Untitled'),
                    'author': article.get('author', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', datetime.now()).isoformat() if isinstance(article.get('published_date'), datetime) else str(article.get('published_date', '')),
                    'summary': article.get('summary', ''),
                    'tags': article.get('tags', []),
                    'reading_time': article.get('reading_time', 5),
                    'engagement_score': round(article.get('engagement_score', 0), 1),
                    'content_preview': article.get('content_preview', ''),
                    'claps': article.get('claps', 0),
                    'member_only': article.get('member_only', False),
                    'publication': article.get('publication', '')
                }
                formatted_articles.append(formatted_article)
            
            return jsonify({
                'success': True,
                'trending_articles': formatted_articles,
                'total_count': len(formatted_articles),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Error in get_trending_articles: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trending_articles': [],
            'total_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/search', methods=['POST'])
def search_articles():
    """
    Search articles by query using web scraping
    
    Expected JSON payload:
    {
        "query": "artificial intelligence",
        "limit": 20
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'query is required'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'query cannot be empty'
            }), 400
        
        limit = min(data.get('limit', 20), 50)
        
        logger.info(f"Searching for: '{query}' (limit: {limit})")
        
        # Search articles using AI agent
        result = ai_agent.search_articles(query, limit)
        
        if result['success']:
            formatted_articles = []
            for article in result['search_results']:
                formatted_article = {
                    'id': str(uuid.uuid4()),
                    'title': article.get('title', 'Untitled'),
                    'author': article.get('author', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_date': article.get('published_date', datetime.now()).isoformat() if isinstance(article.get('published_date'), datetime) else str(article.get('published_date', '')),
                    'summary': article.get('summary', ''),
                    'tags': article.get('tags', []),
                    'reading_time': article.get('reading_time', 5),
                    'ai_score': round(article.get('ai_score', 0), 1),
                    'query_relevance': round(article.get('query_relevance', 0), 1),
                    'content_preview': article.get('content_preview', ''),
                    'claps': article.get('claps', 0),
                    'member_only': article.get('member_only', False),
                    'publication': article.get('publication', '')
                }
                formatted_articles.append(formatted_article)
            
            return jsonify({
                'success': True,
                'search_results': formatted_articles,
                'total_count': len(formatted_articles),
                'query': query,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Error in search_articles: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'search_results': [],
            'total_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/similar', methods=['POST'])
def get_similar_articles():
    """
    Get articles similar to a specific article
    
    Expected JSON payload:
    {
        "article_url": "https://medium.com/...",
        "limit": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'article_url' not in data:
            return jsonify({
                'success': False,
                'error': 'article_url is required'
            }), 400
        
        article_url = data['article_url']
        limit = min(data.get('limit', 5), 20)
        
        # Find the article index in cache
        article_index = -1
        for i, article in enumerate(ai_agent.suggester.articles_cache):
            if article.get('url') == article_url:
                article_index = i
                break
        
        if article_index == -1:
            return jsonify({
                'success': False,
                'error': 'Article not found in cache. Please search for articles first.',
                'suggestion': 'Try using the search endpoint to populate the cache with articles first'
            }), 404
        
        # Get similar articles
        similar_articles = ai_agent.suggester.get_similar_articles(article_index, limit)
        
        formatted_articles = []
        for article in similar_articles:
            formatted_article = {
                'id': str(uuid.uuid4()),
                'title': article.get('title', 'Untitled'),
                'author': article.get('author', 'Unknown'),
                'url': article.get('url', ''),
                'published_date': article.get('published_date', datetime.now()).isoformat() if isinstance(article.get('published_date'), datetime) else str(article.get('published_date', '')),
                'summary': article.get('summary', ''),
                'tags': article.get('tags', []),
                'reading_time': article.get('reading_time', 5),
                'similarity_score': round(article.get('similarity_score', 0), 3),
                'content_preview': article.get('content_preview', ''),
                'claps': article.get('claps', 0),
                'member_only': article.get('member_only', False),
                'publication': article.get('publication', '')
            }
            formatted_articles.append(formatted_article)
        
        return jsonify({
            'success': True,
            'similar_articles': formatted_articles,
            'total_count': len(formatted_articles),
            'original_article_url': article_url,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_similar_articles: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'similar_articles': [],
            'total_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/article/details', methods=['POST'])
def get_article_details():
    """
    Get detailed information about a specific article by scraping
    
    Expected JSON payload:
    {
        "article_url": "https://medium.com/..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'article_url' not in data:
            return jsonify({
                'success': False,
                'error': 'article_url is required'
            }), 400
        
        article_url = data['article_url']
        
        # Validate URL
        if not article_url.startswith('http'):
            return jsonify({
                'success': False,
                'error': 'Invalid article URL'
            }), 400
        
        logger.info(f"Getting details for article: {article_url}")
        
        # Get article details from AI agent
        result = ai_agent.get_article_details(article_url)
        
        if result['success']:
            article = result['article']
            formatted_article = {
                'id': str(uuid.uuid4()),
                'title': article.get('title', 'Untitled'),
                'author': article.get('author', 'Unknown'),
                'url': article.get('url', ''),
                'published_date': article.get('published_date', datetime.now()).isoformat() if isinstance(article.get('published_date'), datetime) else str(article.get('published_date', '')),
                'summary': article.get('summary', ''),
                'content': article.get('content', ''),
                'tags': article.get('tags', []),
                'reading_time': article.get('reading_time', 5),
                'claps': article.get('claps', 0),
                'responses': article.get('responses', 0),
                'engagement_score': round(article.get('engagement_score', 0), 1),
                'member_only': article.get('member_only', False),
                'publication': article.get('publication', ''),
                'scraped_at': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'article': formatted_article,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Error in get_article_details: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/user/interaction', methods=['POST'])
def track_user_interaction():
    """
    Track user interaction with articles for better recommendations
    
    Expected JSON payload:
    {
        "user_id": "user_123",
        "article_url": "https://medium.com/...",
        "action": "read|like|dislike|save|share"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        required_fields = ['user_id', 'article_url', 'action']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        user_id = data['user_id']
        article_url = data['article_url']
        action = data['action']
        
        # Validate action
        valid_actions = ['read', 'like', 'dislike', 'save', 'share', 'bookmark']
        if action not in valid_actions:
            return jsonify({
                'success': False,
                'error': f'Invalid action. Must be one of: {", ".join(valid_actions)}'
            }), 400
        
        logger.info(f"Tracking {action} for user {user_id[:8]}... on article: {article_url[:50]}...")
        
        # Track interaction using AI agent
        result = ai_agent.track_user_interaction(user_id, article_url, action)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in track_user_interaction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/topics', methods=['GET'])
def get_available_topics():
    """Get list of available topics for filtering"""
    topics = [
        'AI', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Neural Networks',
        'Programming', 'JavaScript', 'Python', 'React', 'Node.js', 'Web Development',
        'Data Science', 'Analytics', 'Big Data', 'Data Analysis', 'Statistics',
        'Startup', 'Entrepreneurship', 'Business', 'Product Management', 'Leadership',
        'Design', 'UX', 'UI', 'Product Design', 'User Experience',
        'Technology', 'Software Development', 'Mobile Development', 'iOS', 'Android',
        'DevOps', 'Cloud Computing', 'AWS', 'Docker', 'Kubernetes',
        'Blockchain', 'Cryptocurrency', 'Web3', 'DeFi', 'NFT',
        'Marketing', 'Digital Marketing', 'Content Marketing', 'SEO', 'Social Media',
        'Psychology', 'Productivity', 'Self Improvement', 'Career', 'Remote Work',
        'Health', 'Fitness', 'Mental Health', 'Wellness', 'Meditation',
        'Finance', 'Investing', 'Personal Finance', 'Economics', 'Trading'
    ]
    
    return jsonify({
        'success': True,
        'topics': sorted(topics),
        'total_count': len(topics),
        'categories': {
            'Technology': ['AI', 'Programming', 'Web Development', 'Mobile Development', 'DevOps'],
            'Data': ['Data Science', 'Machine Learning', 'Analytics', 'Big Data'],
            'Business': ['Startup', 'Entrepreneurship', 'Marketing', 'Leadership'],
            'Design': ['UX', 'UI', 'Product Design'],
            'Finance': ['Blockchain', 'Cryptocurrency', 'Investing', 'Personal Finance'],
            'Personal': ['Productivity', 'Health', 'Career', 'Self Improvement']
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/user/preferences', methods=['POST'])
def save_user_preferences():
    """
    Save user preferences for future recommendations
    
    Expected JSON payload:
    {
        "user_id": "user_123",
        "preferences": {
            "topics": ["AI", "Programming"],
            "reading_time": "medium",
            "recency": "week",
            "difficulty": "intermediate"
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        preferences = data.get('preferences', {})
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        # Store user preferences
        user_sessions[user_id] = user_sessions.get(user_id, {})
        user_sessions[user_id]['preferences'] = preferences
        user_sessions[user_id]['updated_at'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': 'Preferences saved successfully',
            'user_id': user_id,
            'preferences': preferences,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saving user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/user/preferences/<user_id>', methods=['GET'])
def get_user_preferences(user_id):
    """Get saved user preferences"""
    try:
        if user_id in user_sessions and 'preferences' in user_sessions[user_id]:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'preferences': user_sessions[user_id]['preferences'],
                'updated_at': user_sessions[user_id].get('updated_at'),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No preferences found for this user',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }), 404
    
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics and performance metrics"""
    try:
        cache_size = len(ai_agent.suggester.articles_cache)
        user_count = len(user_sessions)
        
        # Calculate cache hit ratio (simplified)
        total_requests = ai_agent.suggester.scraper.request_count
        
        return jsonify({
            'success': True,
            'stats': {
                'cached_articles': cache_size,
                'active_users': user_count,
                'total_scraping_requests': total_requests,
                'system_uptime': datetime.now().isoformat(),
                'features_active': [
                    'web_scraping',
                    'ai_scoring',
                    'personalization',
                    'similarity_matching',
                    'user_tracking'
                ]
            },
            'performance': {
                'avg_response_time': '~2-5 seconds',
                'cache_efficiency': 'dynamic',
                'scraping_rate': f'{ai_agent.suggester.scraper.min_delay}-{ai_agent.suggester.scraper.max_delay}s delay'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health - Health check',
            'POST /api/suggestions - Get AI suggestions', 
            'GET /api/trending - Get trending articles',
            'POST /api/search - Search articles',
            'POST /api/similar - Get similar articles',
            'POST /api/article/details - Get article details',
            'POST /api/user/interaction - Track user interaction',
            'GET /api/topics - Get available topics',
            'POST /api/user/preferences - Save user preferences',
            'GET /api/user/preferences/<user_id> - Get user preferences',
            'GET /api/stats - Get system statistics'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded. Please try again later.',
        'retry_after': '60 seconds',
        'timestamp': datetime.now().isoformat()
    }), 429

# Configuration for different environments
class Config:
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size

if __name__ == '__main__':
    config = Config()
    app.config.from_object(config)
    
    print("🚀 Starting Medium AI Content Curator API (Web Scraping)...")
    print(f"📡 Server running on http://{config.HOST}:{config.PORT}")
    print("🕷️  Web scraping mode - No API token required!")
    print("")
    print("📋 Available endpoints:")
    print("   GET  /health                           - Health check")
    print("   POST /api/suggestions                  - Get AI-powered suggestions")
    print("   GET  /api/trending                     - Get trending articles")
    print("   POST /api/search                       - Search articles")
    print("   POST /api/similar                      - Get similar articles")
    print("   POST /api/article/details              - Get detailed article info")
    print("   POST /api/user/interaction             - Track user interactions")
    print("   GET  /api/topics                       - Get available topics")
    print("   POST /api/user/preferences             - Save user preferences")
    print("   GET  /api/user/preferences/<user_id>   - Get user preferences")
    print("   GET  /api/stats                        - Get system statistics")
    print("")
    print("⚡ Features enabled:")
    print("   ✅ AI-powered article scoring")
    print("   ✅ Web scraping from Medium")
    print("   ✅ Personalized recommendations")
    print("   ✅ Content similarity matching")
    print("   ✅ User interaction tracking")
    print("   ✅ Rate limiting & anti-detection")
    print("")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )