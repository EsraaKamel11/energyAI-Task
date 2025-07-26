from fastapi import Header, HTTPException
import logging


class APIKeyAuth:
    def __init__(self, valid_keys=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.valid_keys = valid_keys or {"test-key"}

    def verify_key(self, api_key: str = Header(...)):
        if api_key not in self.valid_keys:
            self.logger.warning(f"Unauthorized API key: {api_key}")
            raise HTTPException(status_code=401, detail="Invalid API Key")
        self.logger.info(f"API key authorized: {api_key}")
        return api_key
