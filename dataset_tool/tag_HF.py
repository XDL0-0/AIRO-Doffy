from huggingface_hub import HfApi

api = HfApi()
repo_id = "IXDLI/wipeBoard_official_unfiltered"

# Most LeRobot datasets use version v2.0
try:
    api.create_tag(repo_id, tag="v3.0", repo_type="dataset")
    print(f"Successfully added v3.0 tag to {repo_id}")
except Exception as e:
    print(f"Error processing tag: {e}")