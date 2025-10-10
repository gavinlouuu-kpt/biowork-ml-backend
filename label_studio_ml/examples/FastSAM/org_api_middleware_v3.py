import os
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    from ls_ml_backend_SAM_middleware import OrganizationAPIMiddlewareV3 as _OrgV3  # type: ignore
    _EXTERNAL = True
    logger.debug("Successfully imported OrganizationAPIMiddlewareV3 from ls_ml_backend_SAM_middleware")
except Exception as e:
    _EXTERNAL = False
    logger.debug(f"Failed to import OrganizationAPIMiddlewareV3: {e}")

_middleware_instance = None


def get_middleware():
    global _middleware_instance
    if _middleware_instance is not None:
        return _middleware_instance
    if _EXTERNAL:
        try:
            _middleware_instance = _OrgV3()
            logger.debug(f"Initialized OrganizationAPIMiddlewareV3: {_middleware_instance}")
            return _middleware_instance
        except Exception as e:
            logger.error(f"Failed to initialize OrganizationAPIMiddlewareV3: {e}")
    # Fallback: lightweight shim that returns env-based creds
    logger.debug("Using fallback shim for credentials (env-based)")
    class _Shim:
        def get_credentials_for_project(self, project_id: int):
            return os.getenv('LABEL_STUDIO_HOST'), os.getenv('LABEL_STUDIO_API_KEY'), 'legacy'

        def get_credentials_from_project_uid(self, project_uid: str):
            return os.getenv('LABEL_STUDIO_HOST'), os.getenv('LABEL_STUDIO_API_KEY'), 'legacy'

    _middleware_instance = _Shim()
    return _middleware_instance


def get_credentials_for_task(task: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        middleware = get_middleware()
        if 'project' in task:
            project_id = task['project']
            logger.debug(f"Getting credentials for project: {project_id}")
            if isinstance(project_id, str):
                return middleware.get_credentials_from_project_uid(project_id)
            return middleware.get_credentials_for_project(project_id)
    except Exception as e:
        logger.error(f"Error getting credentials for task: {e}")
        import traceback
        traceback.print_exc()
    return os.getenv('LABEL_STUDIO_HOST'), os.getenv('LABEL_STUDIO_API_KEY'), 'legacy'


