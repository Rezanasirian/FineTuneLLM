"""
Model Registry for managing trained models and their metadata
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil

from src.core.logger import get_logger

logger = get_logger("model_registry")


class ModelRegistry:
    """Manages registration, versioning, and retrieval of trained models."""

    def __init__(self, registry_dir: Path ):
        """
        Initialize the model registry.

        Args:
            registry_dir (str): Directory to store model registry data.
        """
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self._initialize_registry()
        # logger = logger
        logger.info("Model registry initialized")

    def _initialize_registry(self):
        """Create necessary directories for the registry."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry metadata file if it doesn't exist
        self.registry_file = self.registry_dir / "registry.json"
        if not self.registry_file.exists():
            with open(self.registry_file, "w") as f:
                json.dump({"models": {}}, f, indent=2)
            logger.info(f"Created registry file at {self.registry_file}")

    def _generate_version(self) -> str:
        """Generate a new version string based on timestamp and existing versions."""
        with open(self.registry_file, "r") as f:
            registry_data = json.load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_version = f"v_{timestamp}"
        existing_versions = registry_data["models"].keys()

        # Handle version collision by appending a counter if needed
        counter = 1
        version = base_version
        while version in existing_versions:
            version = f"{base_version}_{counter}"
            counter += 1

        return version

    def register_model(
            self,
            model_path: str,
            metrics: Dict[str, Any],
            config: Dict[str, Any],
            training_results: Dict[str, Any]
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model_path (str): Path to the trained model directory.
            metrics (Dict[str, Any]): Evaluation metrics of the model.
            config (Dict[str, Any]): Training configuration.
            training_results (Dict[str, Any]): Training results (e.g., loss).

        Returns:
            str: Version string of the registered model.
        """
        try:
            # Generate version
            version = self._generate_version()
            logger.info(f"Registering model with version {version}")

            # Create version-specific model directory
            versioned_model_dir = self.models_dir / version
            versioned_model_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files to registry
            src_path = Path(model_path)
            if src_path.exists():
                shutil.copytree(src_path, versioned_model_dir, dirs_exist_ok=True)
                logger.info(f"Copied model files from {src_path} to {versioned_model_dir}")
            else:
                raise ValueError(f"Model path {src_path} does not exist")

            # Prepare metadata
            metadata = {
                "version": version,
                "model_path": str(versioned_model_dir),
                "registration_time": datetime.now().isoformat(),
                "metrics": metrics,
                "config": config,
                "training_results": training_results
            }

            # Save metadata
            metadata_file = self.metadata_dir / f"{version}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved model metadata to {metadata_file}")

            # Update registry
            with open(self.registry_file, "r+") as f:
                registry_data = json.load(f)
                registry_data["models"][version] = {
                    "metadata_file": str(metadata_file),
                    "model_path": str(versioned_model_dir),
                    "registration_time": metadata["registration_time"]
                }
                f.seek(0)
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
                f.truncate()

            logger.info(f"Model {version} successfully registered")
            return version

        except Exception as e:
            logger.error(f"Failed to register model: {e}", exc_info=True)
            raise

    def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific model version.

        Args:
            version (str): Model version to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Model metadata if found, None otherwise.
        """
        metadata_file = self.metadata_dir / f"{version}.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            logger.info(f"Retrieved metadata for model version {version}")
            return metadata
        else:
            logger.warning(f"No metadata found for version {version}")
            return None

    def list_models(self) -> Dict[str, Any]:
        """
        List all registered models.

        Returns:
            Dict[str, Any]: Dictionary of all registered models and their metadata.
        """
        with open(self.registry_file, "r") as f:
            registry_data = json.load(f)
        logger.info("Listed all registered models")
        return registry_data["models"]

    def delete_model(self, version: str) -> bool:
        """
        Delete a model and its metadata from the registry.

        Args:
            version (str): Model version to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            # Check if version exists
            with open(self.registry_file, "r") as f:
                registry_data = json.load(f)

            if version not in registry_data["models"]:
                logger.warning(f"Version {version} not found in registry")
                return False

            # Delete model directory
            model_path = Path(registry_data["models"][version]["model_path"])
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Deleted model directory {model_path}")

            # Delete metadata file
            metadata_file = Path(registry_data["models"][version]["metadata_file"])
            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Deleted metadata file {metadata_file}")

            # Update registry
            del registry_data["models"][version]
            with open(self.registry_file, "w") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Model version {version} successfully deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model version {version}: {e}", exc_info=True)
            return False

    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for the latest registered model.

        Returns:
            Optional[Dict[str, Any]]: Metadata of the latest model, None if no models exist.
        """
        models = self.list_models()
        if not models:
            logger.warning("No models found in registry")
            return None

        # Sort by registration time to get the latest
        latest_version = max(
            models.items(),
            key=lambda x: x[1]["registration_time"],
            default=None
        )[0]
        return self.get_model_info(latest_version)


# if __name__ == "__main__":
#     # Example usage
#     registry = ModelRegistry(registry_dir=Path("../../model_registry"))
#
#     # Example model registration
#     model_path = "D:/Nasirian/projects/FineTuneLLM/model"
#     metrics = {"accuracy": 0.95, "loss": 0.12}
#     config = {"model_name": "example-model", "epochs": 3}
#     training_results = {"final_loss": 0.12}
#
#     version = registry.register_model(model_path, metrics, config, training_results)
#     print(f"Registered model version: {version}")
#
#     # List all models
#     print("All models:", registry.list_models())
#
#     # Get latest model
#     latest_model = registry.get_latest_model()
#     print("Latest model:", latest_model)
#
#     # Get specific model info
#     model_info = registry.get_model_info(version)
#     print(f"Model {version} info:", model_info)