#!/usr/bin/env python3
"""
scripts/orchestrator.py
Main Orchestration Script for Multi-Agent LLM Platform

This script manages the complete lifecycle:
1. Initial training and deployment
2. Continuous monitoring
3. Feedback collection
4. Automated retraining every 2-3 weeks
5. Model version management

Usage:
    python orchestrator.py --mode initial    # Initial setup and training
    python orchestrator.py --mode production # Run production services
    python orchestrator.py --mode retrain    # Trigger retraining
    python orchestrator.py --mode status     # Check system status
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.services.model_registry import ModelRegistry
from src.services.feedback_service import FeedbackDatabase
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.retraining_pipeline import RetrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PlatformOrchestrator:
    """Orchestrates the entire MLOps platform."""

    def __init__(self, config_dir: str = "config"):
        """Initialize orchestrator."""
        self.config = get_config(config_dir)
        self.registry = ModelRegistry(registry_dir=self.config.model_registry)
        self.feedback_db = FeedbackDatabase()

        logger.info("Platform Orchestrator initialized")

    def initial_setup(self):
        """
        Initial platform setup:
        1. Generate training data
        2. Train first model
        3. Deploy API
        """
        logger.info("=" * 80)
        logger.info("INITIAL PLATFORM SETUP")
        logger.info("=" * 80)

        try:
            # Step 1: Run training pipeline
            logger.info("\n[PHASE 1/3] Training initial model...")
            pipeline = TrainingPipeline()
            results = pipeline.run(
                use_existing_data=False,
                skip_evaluation=False,
                register_model=True,
            )

            if results["status"] != "SUCCESS":
                logger.error("Training failed!")
                return False

            logger.info(f"✓ Model version {results['model_version']} trained successfully")

            # Step 2: Deploy API
            logger.info("\n[PHASE 2/3] Deploying API service...")
            self._deploy_api()

            # Step 3: Start scheduler
            logger.info("\n[PHASE 3/3] Starting retraining scheduler...")
            self._start_scheduler()

            logger.info("\n" + "=" * 80)
            logger.info("✓ INITIAL SETUP COMPLETE")
            logger.info("=" * 80)
            logger.info(f"\nAPI running at: http://localhost:{self.config.deployment.api_port}")
            logger.info(f"Model version: {results['model_version']}")
            logger.info("\nNext steps:")
            logger.info("  1. Test API: curl http://localhost:8000/health")
            logger.info("  2. Monitor: http://localhost:3000 (Grafana)")
            logger.info("  3. Check status: python orchestrator.py --mode status")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Initial setup failed: {e}", exc_info=True)
            return False

    def run_production(self):
        """
        Run production services:
        - API server
        - Retraining scheduler
        - Monitoring stack
        """
        logger.info("Starting production services...")

        # Check if Docker is available
        if self._check_docker():
            logger.info("Using Docker Compose for deployment")
            self._run_docker_compose()
        else:
            logger.info("Using systemd for deployment")
            self._run_systemd()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False

    def _run_docker_compose(self):
        """Run services using Docker Compose."""
        compose_file = Path("deployment/docker/docker-compose.yml")

        if not compose_file.exists():
            logger.error(f"Docker Compose file not found: {compose_file}")
            return

        # Start services
        subprocess.run([
            "docker-compose",
            "-f", str(compose_file),
            "up", "-d"
        ])

        logger.info("✓ Services started with Docker Compose")
        logger.info("\nRunning services:")
        subprocess.run(["docker-compose", "-f", str(compose_file), "ps"])

    def _run_systemd(self):
        """Run services using systemd."""
        services = ["llm-api", "llm-scheduler"]

        for service in services:
            subprocess.run(["sudo", "systemctl", "start", f"{service}.service"])
            subprocess.run(["sudo", "systemctl", "enable", f"{service}.service"])
            logger.info(f"✓ Started {service}")

    def _deploy_api(self):
        """Deploy API service."""
        # For simplicity, we'll just start it as a background process
        # In production, use Docker or systemd
        logger.info("Deploying API (use Docker/systemd in production)")

        # Create startup script
        startup_script = Path("start_api.sh")
        with open(startup_script, "w") as f:
            f.write(f"""#!/bin/bash
                        cd {Path.cwd()}
                        source venv/bin/activate 2>/dev/null || true
                        nohup python -m src.api.app > logs/api.log 2>&1 &
                        echo $! > logs/api.pid
                        echo "API started with PID $(cat logs/api.pid)"
                        """)

        startup_script.chmod(0o755)
        subprocess.run(["bash", str(startup_script)])

        logger.info("✓ API deployed")

    def _start_scheduler(self):
        """Start retraining scheduler."""
        logger.info("Starting scheduler (use Docker/systemd in production)")

        startup_script = Path("start_scheduler.sh")
        with open(startup_script, "w") as f:
            f.write(f"""#!/bin/bash
                        cd {Path.cwd()}
                        source venv/bin/activate 2>/dev/null || true
                        nohup python -m src.pipeline.retraining_pipeline > logs/scheduler.log 2>&1 &
                        echo $! > logs/scheduler.pid
                        echo "Scheduler started with PID $(cat logs/scheduler.pid)"
                        """)

        startup_script.chmod(0o755)
        subprocess.run(["bash", str(startup_script)])

        logger.info("✓ Scheduler started")

    def trigger_retraining(self):
        """Manually trigger retraining."""
        logger.info("Triggering manual retraining...")

        pipeline = RetrainingPipeline()
        result = pipeline.run_retraining()

        logger.info(f"\nRetraining result: {result['status']}")
        if result["status"] == "success":
            logger.info(f"New version: {result.get('new_version')}")
            logger.info(f"Duration: {result.get('duration_hours', 0):.2f} hours")

        return result

    def show_status(self):
        """Show current system status."""
        print("\n" + "=" * 80)
        print("MULTI-AGENT LLM PLATFORM STATUS")
        print("=" * 80)

        # Model status
        print("\n[MODEL STATUS]")
        latest = self.registry.get_latest_version()
        if latest:
            print(f"  Current Version: {latest.version}")
            print(f"  Created: {latest.created_at}")
            print(f"  Overall Score: {latest.metrics.get('overall_score', 'N/A')}%")
            print(f"  Status: {latest.status}")
        else:
            print("  No models registered")

        # Feedback statistics
        print("\n[FEEDBACK STATISTICS (Last 14 days)]")
        stats = self.feedback_db.get_feedback_stats(days=14)
        print(f"  Total Feedback: {stats['total_feedback']}")
        print(f"  Average Rating: {stats['average_rating']}/5.0")
        print(f"  Error Rate: {stats['error_rate']}%")
        print(f"  New Training Samples: {stats['new_training_samples']}")

        # Service status
        print("\n[SERVICE STATUS]")
        self._check_service_status()

        # Next retraining
        print("\n[RETRAINING SCHEDULE]")
        if latest:
            created = datetime.fromisoformat(latest.created_at)
            interval = self.config.feedback.retraining_interval_days
            next_retrain = created + timedelta(days=interval)
            days_until = (next_retrain - datetime.now()).days

            print(f"  Last Training: {created.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Next Training: {next_retrain.strftime('%Y-%m-%d %H:%M')} ({days_until} days)")
            print(f"  Schedule: Every {interval} days")

        # Recommendations
        print("\n[RECOMMENDATIONS]")
        decision = RetrainingPipeline().should_retrain()
        if decision["should_retrain"]:
            print("  ⚠️  Retraining recommended:")
            for reason in decision["reasons"]:
                print(f"      - {reason}")
        else:
            print("  ✓ System running optimally")

        print("\n" + "=" * 80)

    def _check_service_status(self):
        """Check if services are running."""
        # Check API
        try:
            import requests
            response = requests.get(f"http://localhost:{self.config.deployment.api_port}/health", timeout=5)
            if response.status_code == 200:
                print("  API: ✓ Running")
            else:
                print(f"  API: ✗ Not responding (status {response.status_code})")
        except:
            print("  API: ✗ Not running")

        # Check scheduler (check if PID file exists and process is running)
        pid_file = Path("logs/scheduler.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process exists
                os.kill(pid, 0)
                print(f"  Scheduler: ✓ Running (PID {pid})")
            except:
                print("  Scheduler: ✗ Not running")
        else:
            print("  Scheduler: ✗ Not running")

    def stop_services(self):
        """Stop all services."""
        logger.info("Stopping services...")

        # Stop API
        pid_file = Path("logs/api.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 15)  # SIGTERM
                pid_file.unlink()
                logger.info("✓ API stopped")
            except:
                logger.warning("Could not stop API")

        # Stop scheduler
        pid_file = Path("logs/scheduler.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 15)
                pid_file.unlink()
                logger.info("✓ Scheduler stopped")
            except:
                logger.warning("Could not stop scheduler")

        logger.info("Services stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent LLM Platform Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Initial setup
              python orchestrator.py --mode initial
            
              # Run production services
              python orchestrator.py --mode production
            
              # Trigger retraining
              python orchestrator.py --mode retrain
            
              # Check status
              python orchestrator.py --mode status
            
              # Stop services
              python orchestrator.py --mode stop
                    """
                )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["initial", "production", "retrain", "status", "stop"],
        help="Operation mode",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory",
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = PlatformOrchestrator(config_dir=args.config_dir)

    # Execute based on mode
    if args.mode == "initial":
        success = orchestrator.initial_setup()
        sys.exit(0 if success else 1)

    elif args.mode == "production":
        orchestrator.run_production()

    elif args.mode == "retrain":
        result = orchestrator.trigger_retraining()
        sys.exit(0 if result["status"] == "success" else 1)

    elif args.mode == "status":
        orchestrator.show_status()

    elif args.mode == "stop":
        orchestrator.stop_services()


if __name__ == "__main__":
    main()
