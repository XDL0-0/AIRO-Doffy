from huggingface_hub import HfApi
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

api = HfApi()
repo_id = "IXDLI/wipeBoard_official_unfiltered"

# Most LeRobot datasets use version v2.0
try:
    api.create_tag(repo_id, tag="v3.0", repo_type="dataset")
    logger.info(f"Successfully added v3.0 tag to {repo_id}")
except Exception as e:
    logger.error(f"Error processing tag: {e}")
