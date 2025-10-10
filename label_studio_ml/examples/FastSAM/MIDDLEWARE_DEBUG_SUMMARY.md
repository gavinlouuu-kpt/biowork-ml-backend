# Middleware Debug Summary

## Issue
The ML backend was failing to acquire API keys for accessing Label Studio uploaded files, resulting in the error:
```
FileNotFoundError: To access uploaded and local storage files you have to set LABEL_STUDIO_API_KEY environment variable.
```

## Root Cause
The middleware integration was not working because:

1. **Missing Implementation**: The `org_api_middleware_v3.py` file was only a shim that tried to import from `ls_ml_backend_SAM_middleware`, which didn't exist in the Docker container.

2. **Fallback Behavior**: When the import failed, it fell back to reading from environment variables (`LABEL_STUDIO_API_KEY`), which were not set.

3. **Silent Failure**: The exception was caught silently, so no error was logged about the missing middleware implementation.

## Solution

### 1. Copy Real Middleware Implementation
Copied the actual `OrganizationAPIMiddlewareV3` implementation from `/home/gavin/code/ls-ml-backend-SAM/org_api_middleware_v3.py` to the segment_anything_2_image directory as `ls_ml_backend_SAM_middleware.py`.

### 2. Enhanced Logging
Added debug logging to track:
- Middleware import success/failure
- Middleware initialization
- Credential acquisition process
- Token retrieval from database

### 3. Database Configuration
The middleware was already configured in `.env` to use PostgreSQL:
```bash
LABEL_STUDIO_DB_TYPE=postgres
LABEL_STUDIO_DB_HOST=db
LABEL_STUDIO_DB_PORT=5432
LABEL_STUDIO_DB_NAME=postgres
LABEL_STUDIO_DB_USER=postgres
LABEL_STUDIO_DB_PASSWORD=postgres
```

## Verification

After the fix, the logs show successful operation:

```
[INFO] Successfully imported OrganizationAPIMiddlewareV3 from ls_ml_backend_SAM_middleware
[INFO] Organization API Middleware V3 initialized with PostgreSQL: postgres@db:5432/postgres
[DEBUG] Project 39 belongs to organization 1
[DEBUG] Organization 1 JWT settings: {'api_tokens_enabled': True, 'legacy_api_tokens_enabled': True}
[DEBUG] Retrieved legacy token for organization 1
[INFO] Using legacy token for organization 1
[DEBUG] Middleware credentials acquired: hostname=http://label-studio-app:8000, token=***
[INFO] File downloaded to /root/.cache/label-studio/8ad2ca74__79712951-test_02551_Cam_V710_Cine1.png
[INFO] Image embeddings computed.
[DEBUG] Response status: 200 OK
```

## Key Files Modified

1. **ls_ml_backend_SAM_middleware.py** (new): Real middleware implementation with PostgreSQL support
2. **org_api_middleware_v3.py**: Shim that imports and initializes the real middleware
3. **model.py**: Uses middleware to get credentials before calling `get_local_path()`

## How It Works

1. When a prediction request comes in, `model.py` calls `org_api_middleware_v3.get_credentials_for_task(task)`
2. The middleware extracts the project ID from the task
3. It queries the PostgreSQL database to:
   - Find the organization for the project
   - Get JWT settings for the organization
   - Retrieve the legacy API token for the organization's creator
4. Returns `(hostname, api_token, token_type)` tuple
5. These credentials are passed to `get_local_path()` which uses them to download the image from Label Studio

## Benefits

- **Automatic token resolution**: No need to manually configure API keys
- **Multi-organization support**: Each organization's tokens are automatically retrieved
- **Fallback support**: Can fall back to environment variables if database is unavailable
- **Token caching**: Improves performance by caching database queries
- **JWT support**: Handles both legacy tokens and JWT Personal Access Tokens

