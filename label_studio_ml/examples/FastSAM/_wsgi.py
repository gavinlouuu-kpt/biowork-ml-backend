import os
import argparse
import logging
import logging.config
import json

# Load .env
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        load_dotenv()
except Exception:
    pass

# Initialize organization middleware V3 if enabled
USE_ORG_MIDDLEWARE = os.getenv('USE_ORG_MIDDLEWARE', 'false').lower() in ('true', '1', 'yes')
if USE_ORG_MIDDLEWARE:
    try:
        from org_api_middleware_v3 import OrganizationAPIMiddlewareV3
        db_type = os.getenv('LABEL_STUDIO_DB_TYPE', '').lower()
        ls_host = os.getenv('LABEL_STUDIO_HOST')
        db_host = os.getenv('LABEL_STUDIO_DB_HOST')
        db_port = os.getenv('LABEL_STUDIO_DB_PORT')
        db_name = os.getenv('LABEL_STUDIO_DB_NAME')
        db_user = os.getenv('LABEL_STUDIO_DB_USER')
        db_password = os.getenv('LABEL_STUDIO_DB_PASSWORD')
        db_path = os.getenv('LABEL_STUDIO_DB_PATH')
        if not db_type:
            if db_host and db_name:
                db_type = 'postgres'
            elif db_path:
                db_type = 'sqlite'
        if ls_host and (db_type in ('postgres', 'sqlite')):
            _ = OrganizationAPIMiddlewareV3(
                db_path=db_path,
                label_studio_host=ls_host,
                db_type=db_type,
                db_host=db_host,
                db_port=int(db_port) if db_port else None,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password
            )
            print("✅ Organization middleware V3 enabled")
        else:
            print("⚠️  USE_ORG_MIDDLEWARE=true but missing required environment variables")
            USE_ORG_MIDDLEWARE = False
    except Exception as e:
        print(f"⚠️  Failed to initialize organization middleware: {e}")
        USE_ORG_MIDDLEWARE = False
else:
    print("ℹ️  Organization middleware disabled. Using static token authentication.")

logging.config.dictConfig({
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {"standard": {"format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"}},
  "handlers": {"console": {"class": "logging.StreamHandler", "level": os.getenv('LOG_LEVEL', 'INFO'), "stream": "ext://sys.stdout", "formatter": "standard"}},
  "root": {"level": os.getenv('LOG_LEVEL', 'INFO'), "handlers": ["console"], "propagate": True}
})

from label_studio_ml.api import init_app
from model import SamMLBackend

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label Studio FastSAM')
    parser.add_argument('-p', '--port', dest='port', type=int, default=9090, help='Server port')
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='), help='Additional model kwargs')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Debug mode')
    parser.add_argument('--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None, help='Logging level')
    parser.add_argument('--check', dest='check', action='store_true', help='Validate model creation')
    parser.add_argument('--basic-auth-user', default=os.environ.get('ML_SERVER_BASIC_AUTH_USER', None), help='Basic auth user')
    parser.add_argument('--basic-auth-pass', default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS', None), help='Basic auth pass')

    args = parser.parse_args()
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'false':
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()
    if args.kwargs:
        kwargs.update(parse_kwargs())
    if args.check:
        print('Check "' + SamMLBackend.__name__ + '" instance creation..')
        _ = SamMLBackend(**kwargs)
    app = init_app(model_class=SamMLBackend, basic_auth_user=args.basic_auth_user, basic_auth_pass=args.basic_auth_pass)
    app.run(host=args.host, port=args.port, debug=args.debug)
else:
    app = init_app(model_class=SamMLBackend)
