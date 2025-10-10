"""
Organization API Token Middleware V3

This middleware automatically retrieves authentication tokens for Label Studio ML backends.
It supports both legacy tokens and JWT Personal Access Tokens (PATs) with automatic fallback.

Features:
- Automatic token type detection
- Falls back from legacy to PAT if legacy tokens are disabled
- JWT token refresh handling
- Token caching for performance
- No Label Studio configuration changes required
- **Supports both SQLite and PostgreSQL databases**
"""

import os
import logging
import sqlite3
from typing import Optional, Dict, Tuple, Any
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed. PostgreSQL support disabled.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. JWT token refresh disabled.")


class OrganizationAPIMiddlewareV3:
    """
    Enhanced middleware for fetching organization API tokens from Label Studio database.
    
    Supports both legacy tokens and JWT Personal Access Tokens with automatic fallback.
    Works with both SQLite and PostgreSQL databases.
    """
    
    def __init__(
        self, 
        db_path: Optional[str] = None, 
        label_studio_host: Optional[str] = None,
        db_type: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            db_path: Path to Label Studio SQLite database (for SQLite only)
            label_studio_host: Label Studio host URL
            db_type: Database type ('sqlite' or 'postgres'). Auto-detected if not specified.
            db_host: PostgreSQL host (for PostgreSQL only)
            db_port: PostgreSQL port (for PostgreSQL only)
            db_name: PostgreSQL database name (for PostgreSQL only)
            db_user: PostgreSQL username (for PostgreSQL only)
            db_password: PostgreSQL password (for PostgreSQL only)
        """
        self.label_studio_host = label_studio_host or os.environ.get('LABEL_STUDIO_HOST')
        
        if not self.label_studio_host:
            raise ValueError("Label Studio host must be provided either as parameter or LABEL_STUDIO_HOST env var")
        
        # Auto-detect database type
        self.db_type = db_type or os.environ.get('LABEL_STUDIO_DB_TYPE', '').lower()
        
        # PostgreSQL configuration
        self.db_host = db_host or os.environ.get('LABEL_STUDIO_DB_HOST')
        self.db_port = db_port or int(os.environ.get('LABEL_STUDIO_DB_PORT', '5432'))
        self.db_name = db_name or os.environ.get('LABEL_STUDIO_DB_NAME')
        self.db_user = db_user or os.environ.get('LABEL_STUDIO_DB_USER')
        self.db_password = db_password or os.environ.get('LABEL_STUDIO_DB_PASSWORD')
        
        # SQLite configuration
        self.db_path = db_path or os.environ.get(
            'LABEL_STUDIO_DB_PATH',
            '/label-studio/data/label_studio.sqlite3'
        )
        
        # Auto-detect database type if not specified
        if not self.db_type:
            if self.db_host and self.db_name:
                self.db_type = 'postgres'
            elif self.db_path:
                self.db_type = 'sqlite'
            else:
                raise ValueError(
                    "Could not auto-detect database type. "
                    "Provide either db_path (SQLite) or db_host+db_name (PostgreSQL)"
                )
        
        # Validate configuration based on database type
        if self.db_type == 'postgres':
            if not POSTGRES_AVAILABLE:
                raise ImportError(
                    "PostgreSQL support requires psycopg2. "
                    "Install with: pip install psycopg2-binary"
                )
            if not all([self.db_host, self.db_name, self.db_user, self.db_password]):
                raise ValueError(
                    "PostgreSQL requires: LABEL_STUDIO_DB_HOST, LABEL_STUDIO_DB_NAME, "
                    "LABEL_STUDIO_DB_USER, LABEL_STUDIO_DB_PASSWORD"
                )
            logger.info(
                f"Organization API Middleware V3 initialized with PostgreSQL: "
                f"{self.db_user}@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        elif self.db_type == 'sqlite':
            if not self.db_path:
                raise ValueError("SQLite requires LABEL_STUDIO_DB_PATH")
            if not os.path.exists(self.db_path):
                logger.warning(f"Database file not found at {self.db_path}. Token resolution will fail.")
            else:
                logger.info(f"Organization API Middleware V3 initialized with SQLite: {self.db_path}")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}. Use 'sqlite' or 'postgres'")
        
        # Cache for JWT access tokens (short-lived)
        self._jwt_access_token_cache: Dict[int, Tuple[str, float]] = {}
    
    def _get_db_connection(self) -> Any:
        """Get a connection to the Label Studio database."""
        if self.db_type == 'postgres':
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            return conn
        else:  # sqlite
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Database file not found at {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
    
    def _execute_query(self, query: str, params: tuple) -> Optional[Dict]:
        """
        Execute a query and return the first result as a dictionary.
        
        Args:
            query: SQL query (use ? for SQLite, %s for PostgreSQL)
            params: Query parameters
            
        Returns:
            Dictionary with result or None
        """
        conn = self._get_db_connection()
        
        try:
            if self.db_type == 'postgres':
                # Convert ? to %s for PostgreSQL
                query = query.replace('?', '%s')
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            else:
                cursor = conn.cursor()
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            conn.close()
            
            if result:
                if self.db_type == 'postgres':
                    return dict(result)
                else:
                    return dict(result)
            return None
            
        except Exception as e:
            conn.close()
            raise e
    
    @lru_cache(maxsize=128)
    def get_project_organization(self, project_id: int) -> Optional[int]:
        """
        Get the organization ID for a given project.
        
        Args:
            project_id: The project ID
            
        Returns:
            Organization ID or None if not found
        """
        try:
            query = """
                SELECT organization_id 
                FROM project 
                WHERE id = ?
            """
            
            result = self._execute_query(query, (project_id,))
            
            if result:
                org_id = result['organization_id']
                logger.debug(f"Project {project_id} belongs to organization {org_id}")
                return org_id
            else:
                logger.warning(f"Project {project_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching organization for project {project_id}: {e}")
            return None
    
    @lru_cache(maxsize=128)
    def get_jwt_settings(self, organization_id: int) -> Optional[Dict]:
        """
        Get JWT settings for an organization.
        
        Args:
            organization_id: The organization ID
            
        Returns:
            Dictionary with JWT settings or None if not found
        """
        try:
            query = """
                SELECT api_tokens_enabled, legacy_api_tokens_enabled 
                FROM jwt_auth_jwtsettings 
                WHERE organization_id = ?
            """
            
            result = self._execute_query(query, (organization_id,))
            
            if result:
                settings = {
                    'api_tokens_enabled': bool(result['api_tokens_enabled']),
                    'legacy_api_tokens_enabled': bool(result['legacy_api_tokens_enabled'])
                }
                logger.debug(f"Organization {organization_id} JWT settings: {settings}")
                return settings
            else:
                logger.debug(f"No JWT settings found for organization {organization_id}, assuming defaults")
                # Default: both enabled
                return {
                    'api_tokens_enabled': True,
                    'legacy_api_tokens_enabled': True
                }
                
        except Exception as e:
            logger.error(f"Error fetching JWT settings for organization {organization_id}: {e}")
            return None
    
    @lru_cache(maxsize=128)
    def get_organization_admin_legacy_token(self, organization_id: int) -> Optional[str]:
        """
        Get a legacy API token for the organization.
        
        Args:
            organization_id: The organization ID
            
        Returns:
            Legacy API token string or None if not found
        """
        try:
            # Find the organization creator (they always have access)
            query = """
                SELECT created_by_id 
                FROM organization 
                WHERE id = ?
            """
            
            result = self._execute_query(query, (organization_id,))
            
            if not result:
                logger.warning(f"Organization {organization_id} not found")
                return None
            
            user_id = result['created_by_id']
            
            # Get the user's legacy auth token
            token_query = """
                SELECT key 
                FROM authtoken_token 
                WHERE user_id = ?
            """
            
            token_result = self._execute_query(token_query, (user_id,))
            
            if token_result:
                token = token_result['key']
                logger.debug(f"Retrieved legacy token for organization {organization_id}")
                return token
            else:
                logger.debug(f"No legacy auth token found for user {user_id} in organization {organization_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching legacy token for organization {organization_id}: {e}")
            return None
    
    @lru_cache(maxsize=128)
    def get_organization_admin_jwt_refresh_token(self, organization_id: int) -> Optional[str]:
        """
        Get a JWT refresh token (PAT) for the organization.
        
        Args:
            organization_id: The organization ID
            
        Returns:
            JWT refresh token string or None if not found
        """
        try:
            # Find the organization creator
            query = """
                SELECT created_by_id 
                FROM organization 
                WHERE id = ?
            """
            
            result = self._execute_query(query, (organization_id,))
            
            if not result:
                logger.warning(f"Organization {organization_id} not found")
                return None
            
            user_id = result['created_by_id']
            
            # Get the user's JWT refresh token (PAT)
            # Query differs slightly between SQLite and PostgreSQL for datetime comparison
            if self.db_type == 'postgres':
                token_query = """
                    SELECT ot.token, ot.expires_at 
                    FROM token_blacklist_outstandingtoken ot
                    LEFT JOIN token_blacklist_blacklistedtoken bt ON ot.id = bt.token_id
                    WHERE ot.user_id = ? 
                      AND bt.id IS NULL
                      AND ot.expires_at > NOW()
                    ORDER BY ot.created_at DESC
                    LIMIT 1
                """
            else:  # sqlite
                token_query = """
                    SELECT ot.token, ot.expires_at 
                    FROM token_blacklist_outstandingtoken ot
                    LEFT JOIN token_blacklist_blacklistedtoken bt ON ot.id = bt.token_id
                    WHERE ot.user_id = ? 
                      AND bt.id IS NULL
                      AND datetime(ot.expires_at) > datetime('now')
                    ORDER BY ot.created_at DESC
                    LIMIT 1
                """
            
            token_result = self._execute_query(token_query, (user_id,))
            
            if token_result:
                token = token_result['token']
                logger.debug(f"Retrieved JWT refresh token for organization {organization_id}")
                return token
            else:
                logger.debug(f"No JWT refresh token found for user {user_id} in organization {organization_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching JWT refresh token for organization {organization_id}: {e}")
            return None
    
    def _refresh_jwt_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Exchange a JWT refresh token for an access token.
        
        Args:
            refresh_token: The JWT refresh token (PAT)
            
        Returns:
            Access token or None if refresh fails
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available. Cannot refresh JWT token.")
            return None
            
        try:
            import requests
            url = f"{self.label_studio_host}/api/token/refresh/"
            response = requests.post(
                url,
                json={"refresh": refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                access_token = data.get('access')
                logger.debug("Successfully refreshed JWT access token")
                return access_token
            else:
                logger.warning(f"Failed to refresh JWT token: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error refreshing JWT access token: {e}")
            return None
    
    def get_jwt_access_token(self, organization_id: int) -> Optional[str]:
        """
        Get a valid JWT access token for the organization.
        Caches tokens and refreshes when needed.
        
        Args:
            organization_id: The organization ID
            
        Returns:
            JWT access token or None if unavailable
        """
        # Check cache first
        if organization_id in self._jwt_access_token_cache:
            token, expiry = self._jwt_access_token_cache[organization_id]
            # If token expires in more than 60 seconds, use it
            if datetime.now(timezone.utc).timestamp() < expiry - 60:
                logger.debug(f"Using cached JWT access token for organization {organization_id}")
                return token
            else:
                logger.debug(f"Cached JWT access token expired for organization {organization_id}")
        
        # Get refresh token and exchange for access token
        refresh_token = self.get_organization_admin_jwt_refresh_token(organization_id)
        if not refresh_token:
            logger.debug(f"No JWT refresh token available for organization {organization_id}")
            return None
        
        access_token = self._refresh_jwt_access_token(refresh_token)
        if access_token:
            # Cache with 5 minute expiry (access tokens typically last 5 minutes)
            expiry = datetime.now(timezone.utc).timestamp() + 300
            self._jwt_access_token_cache[organization_id] = (access_token, expiry)
            return access_token
        
        return None
    
    def get_credentials_for_project(self, project_id: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Get the Label Studio host and API token for a given project.
        Automatically selects the best available authentication method.
        
        Args:
            project_id: The project ID from the ML backend request
            
        Returns:
            Tuple of (hostname, api_token, token_type) where token_type is 'legacy', 'jwt', or None
        """
        org_id = self.get_project_organization(project_id)
        
        if org_id is None:
            logger.error(f"Could not determine organization for project {project_id}")
            return None, None, None
        
        # Check JWT settings to determine which token type to use
        jwt_settings = self.get_jwt_settings(org_id)
        
        if jwt_settings and jwt_settings.get('legacy_api_tokens_enabled'):
            # Try legacy token first (simpler, no refresh needed)
            token = self.get_organization_admin_legacy_token(org_id)
            if token:
                logger.info(f"Using legacy token for organization {org_id}")
                return self.label_studio_host, token, 'legacy'
        
        # Fall back to JWT PAT
        if jwt_settings and jwt_settings.get('api_tokens_enabled'):
            access_token = self.get_jwt_access_token(org_id)
            if access_token:
                logger.info(f"Using JWT access token for organization {org_id}")
                return self.label_studio_host, access_token, 'jwt'
        
        logger.error(f"Could not retrieve any valid token for organization {org_id}")
        return None, None, None
    
    def get_credentials_from_project_uid(self, project_uid: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Get credentials from a project UID string (format: "project_id.timestamp").
        
        Args:
            project_uid: Project UID from Label Studio (e.g., "123.1609459200")
            
        Returns:
            Tuple of (hostname, api_token, token_type)
        """
        try:
            project_id = int(project_uid.split('.')[0])
            return self.get_credentials_for_project(project_id)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid project_uid format: {project_uid}. Error: {e}")
            return None, None, None
    
    def clear_cache(self):
        """Clear all caches."""
        self.get_project_organization.cache_clear()
        self.get_jwt_settings.cache_clear()
        self.get_organization_admin_legacy_token.cache_clear()
        self.get_organization_admin_jwt_refresh_token.cache_clear()
        self._jwt_access_token_cache.clear()
        logger.info("Middleware cache cleared")


# Global instance to be used by the ML backend
_middleware_instance: Optional[OrganizationAPIMiddlewareV3] = None


def get_middleware() -> OrganizationAPIMiddlewareV3:
    """
    Get or create the global middleware instance.
    
    Returns:
        OrganizationAPIMiddlewareV3 instance
    """
    global _middleware_instance
    
    if _middleware_instance is None:
        _middleware_instance = OrganizationAPIMiddlewareV3()
    
    return _middleware_instance


def get_credentials_for_task(task: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Convenience function to get credentials from a task dictionary.
    
    Args:
        task: Task dictionary from Label Studio
        
    Returns:
        Tuple of (hostname, api_token, token_type)
    """
    middleware = get_middleware()
    
    # Try to extract project info from task
    if 'project' in task:
        project_id = task['project']
        if isinstance(project_id, str):
            return middleware.get_credentials_from_project_uid(project_id)
        else:
            return middleware.get_credentials_for_project(project_id)
    
    logger.warning("No project information found in task")
    return None, None, None
