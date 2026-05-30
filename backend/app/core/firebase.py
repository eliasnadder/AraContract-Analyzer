import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
import logging

logger = logging.getLogger(__name__)


def initialize_firebase(service_account_path: str):
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise


def verify_firebase_token(token: str) -> dict:
    return firebase_auth.verify_id_token(token)
