"""
Model Registry for managing trained models and their metadata
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil
from dataclasses import dataclass

from src.core.logger import get_logger
from src.services.feedback_database import FeedbackDatabase, FeedbackStats

logger = get_logger("model_registry")


@dataclass
class ModelVersion:
    """Data class for model version information."""
    version: str
    model_path: str
    created_at: str
    status: str = "active"
    metrics: Dict[str, Any] = None


class ModelRegistry:
    """Manages registration, versioning, and retrieval of trained models."""

    def __init__(self, registry_dir: Path):
        """
        Initialize the model registry.

        Args:
            registry_dir (str): Directory to store model registry data.
        """
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.feedback_db = FeedbackDatabase()  # Initialize feedback database
        self._initialize_registry()
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
                "training_results": training_results,
                "status": "active"
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
                    "registration_time": metadata["registration_time"],
                    "status": "active"
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

    def list_versions(self) -> List[str]:
        """
        List all model versions.

        Returns:
            List[str]: List of all version strings.
        """
        models = self.list_models()
        return list(models.keys())

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

    def get_latest_version(self) -> Optional[ModelVersion]:
        """
        Get the latest ACTIVE model version as a ModelVersion object.

        Returns:
            Optional[ModelVersion]: Latest active model version, None if no active models exist.
        """
        models = self.list_models()
        if not models:
            logger.warning("No models found in registry")
            return None

        # Filter active models and sort by registration time
        active_models = {
            version: info for version, info in models.items()
            if info.get("status", "active") == "active"
        }

        if not active_models:
            logger.warning("No active models found in registry")
            return None

        # Get the most recent active model
        latest_version_str = max(
            active_models.items(),
            key=lambda x: x[1]["registration_time"]
        )[0]

        # Get full metadata
        metadata = self.get_model_info(latest_version_str)
        if not metadata:
            return None

        # Create ModelVersion object
        model_version = ModelVersion(
            version=metadata["version"],
            model_path=metadata["model_path"],
            created_at=metadata["registration_time"],
            status=metadata.get("status", "active"),
            metrics=metadata.get("metrics", {})
        )

        logger.info(f"Retrieved latest version: {model_version.version}")
        return model_version

    def set_active_version(self, version: str) -> bool:
        """
        Set a specific version as the active version.

        Args:
            version (str): Version to set as active.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get all models
            models = self.list_models()
            if version not in models:
                logger.error(f"Version {version} not found")
                return False

            # Deactivate all other versions
            for v in models:
                if v != version:
                    metadata_file = self.metadata_dir / f"{v}.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        metadata["status"] = "inactive"
                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)

                        # Update registry entry
                        models[v]["status"] = "inactive"

            # Activate target version
            metadata_file = self.metadata_dir / f"{version}.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                metadata["status"] = "active"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                models[version]["status"] = "active"

            # Save updated registry
            with open(self.registry_file, "w") as f:
                json.dump({"models": models}, f, indent=2, ensure_ascii=False)

            logger.info(f"Set version {version} as active")
            return True

        except Exception as e:
            logger.error(f"Failed to set active version {version}: {e}", exc_info=True)
            return False

    def create_backup(self, version: str) -> str:
        """
        Create a backup of a specific model version.

        Args:
            version (str): Version to backup.

        Returns:
            str: Path to backup directory.
        """
        try:
            metadata = self.get_model_info(version)
            if not metadata:
                raise ValueError(f"Version {version} not found")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.registry_dir / f"backups" / f"{version}_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy model files
            model_path = Path(metadata["model_path"])
            shutil.copytree(model_path, backup_dir / "model", dirs_exist_ok=True)

            # Copy metadata
            metadata_backup = metadata.copy()
            metadata_backup["backup_time"] = datetime.now().isoformat()
            with open(backup_dir / "metadata.json", "w") as f:
                json.dump(metadata_backup, f, indent=2, ensure_ascii=False)

            logger.info(f"Created backup of {version} at {backup_dir}")
            return str(backup_dir)

        except Exception as e:
            logger.error(f"Failed to create backup of {version}: {e}", exc_info=True)
            return ""

    def update_model_feedback(self, version: str, feedback_stats: FeedbackStats):
        """
        Update model metrics with feedback data.

        Args:
            version (str): Model version to update.
            feedback_stats (FeedbackStats): Feedback statistics.
        """
        metadata = self.get_model_info(version)
        if metadata:
            # Update metrics with feedback data
            if "metrics" not in metadata:
                metadata["metrics"] = {}

            metadata["metrics"].update({
                'feedback_count': feedback_stats.total_feedback,
                'avg_rating': feedback_stats.average_rating,
                'error_rate': feedback_stats.error_rate,
                'satisfaction_trend': feedback_stats.satisfaction_trend,
                'top_error_types': {k: v for k, v in feedback_stats.top_error_types},
                'last_feedback_update': datetime.now().isoformat()
            })

            # Calculate overall score
            overall_score = round(
                (feedback_stats.average_rating / 5.0 * 40) +
                ((100 - feedback_stats.error_rate) / 100 * 40) +
                (feedback_stats.satisfaction_trend / 10 * 20),  # Normalized
                1
            )
            metadata["metrics"]["overall_score"] = overall_score

            # Save updated metadata
            metadata_file = self.metadata_dir / f"{version}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated feedback metrics for {version}: {overall_score}%")
        else:
            logger.warning(f"Cannot update feedback for unknown version {version}")

    def get_model_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance trend across model versions.

        Args:
            days (int): Days of feedback to analyze.

        Returns:
            Dict[str, Any]: Performance trend data.
        """
        versions = self.list_versions()
        trend_data = {}

        for version in sorted(versions, key=lambda v: v, reverse=True)[:10]:
            model_info = self.get_model_info(version)
            if not model_info:
                continue

            # Get feedback for this version
            feedback = self.feedback_db.get_feedback_by_version(version, limit=1000)

            if feedback:
                avg_rating = sum(f.rating or 0 for f in feedback if f.rating) / len([f for f in feedback if f.rating])
                error_rate = len([f for f in feedback if f.is_error]) / len(feedback) * 100

                trend_data[version] = {
                    'avg_rating': round(avg_rating, 2),
                    'error_rate': round(error_rate, 2),
                    'feedback_count': len(feedback),
                    'deployed_at': model_info['registration_time'],
                    'overall_score': model_info['metrics'].get('overall_score', 0)
                }

        logger.info(f"Generated performance trend for {len(trend_data)} versions")
        return trend_data