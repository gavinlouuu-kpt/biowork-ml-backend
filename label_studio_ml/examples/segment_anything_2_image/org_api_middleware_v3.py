import os
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    from ls_ml_backend_SAM_middleware import OrganizationAPIMiddlewareV3 as _OrgV3  # type: ignore
    _EXTERNAL = True
except Exception:
    _EXTERNAL = False

_middleware_instance = None


def get_middleware():
    global _middleware_instance
    if _middleware_instance is not None:
        return _middleware_instance
    if _EXTERNAL:
        try:
            _middleware_instance = _OrgV3()
            return _middleware_instance
        except Exception:
            pass
    # Fallback: lightweight shim that returns env-based creds
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
            if isinstance(project_id, str):
                return middleware.get_credentials_from_project_uid(project_id)
            return middleware.get_credentials_for_project(project_id)
    except Exception:
        pass
    return os.getenv('LABEL_STUDIO_HOST'), os.getenv('LABEL_STUDIO_API_KEY'), 'legacy'


